# -*- coding: utf-8 -*-
# code by Sampa Misra
# Optional REINFORCE with AMP/MIC scorers. Variable-length generation (EOS).
# Saves CSV/FASTA with AMP prob (high), MIC prob (low), length, hydrophobicity,
# hydrophobic moment, charge (pH 7.4), and isoelectric point.

import os, math, random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Config
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 7
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Alphabet & special tokens
AA = list("ACDEFGHIKLMNPQRSTVWY")            # 20 AAs
AA2IDX = {a: i+1 for i, a in enumerate(AA)}  # 1..20
IDX2AA = {i+1: a for i, a in enumerate(AA)}
PAD_ID = 0
BOS_ID = 21
EOS_ID = 22
VOCAB_SIZE = 23                               # 20 AA + PAD + BOS + EOS

# IMPORTANT: Scorer vocab (your classifiers were trained on PAD+20 only)
SCORER_VOCAB_SIZE = 21                        # PAD(0) + 20 AAs; no BOS/EOS

# Lengths
MAX_CONTENT_LEN = 25                           # max AA content (truncate for training)
MAX_STEPS = MAX_CONTENT_LEN + 1               # include EOS step at inference

# VAE
HIDDEN_DIM  = 128
LATENT_DIM  = 64
RCL_WEIGHT  = 64.0
KL_MIN, KL_MAX, KL_RATE = 1e-4, 1e-2, 0.01

# Train
BATCH = 128
EPOCHS = 100
PRINT_EVERY = 50
LR = 1e-3

# RL fine-tune
DO_RL = True
RL_STEPS = 5000
RL_BATCH = 128
RL_LR = 1e-4
RL_W_AMP = 1.0
RL_W_MIC = 1.0
MIC_POSITIVE_IS_LOW = True    # 1 = low MIC (good)
ENTROPY_W = 0.0               # >0 encourages exploration (e.g., 1e-3)

# Paths
DATA_DIR  = "Dataset/1ststage"
OUT_DIR   = "Result_1ststage"
MODEL_DIR = "Model/Model_1ststage"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

POS_CSV   = os.path.join(DATA_DIR, "unlabelled_positive.csv")
NEG_CSV   = os.path.join(DATA_DIR, "unlabelled_negative.csv")
MIC_CSV   = os.path.join(DATA_DIR, "mic_data.csv")
UNI_TRAIN = os.path.join(DATA_DIR, "Uniprot_0_25_train.csv")
UNI_VAL   = os.path.join(DATA_DIR, "Uniprot_0_25_val.csv")

# Pretrained scorers
AMP_CLS_PT = os.path.join(MODEL_DIR, "amp_classifier.pt")
MIC_CLS_PT = os.path.join(MODEL_DIR, "mic_classifier_best.pt")  # fixed single MIC model

# Save points
CKPT_PATH = os.path.join(MODEL_DIR, "VAE_ECD_RL_best.ckpt")
ENC_PATH  = os.path.join(MODEL_DIR, "VAE_ECD_RL_encoder.pt")
DEC_PATH  = os.path.join(MODEL_DIR, "VAE_ECD_RL_decoder.pt")

# ================= Physicochemical property helpers =================
KD = {
    "A":1.8,"C":2.5,"D":-3.5,"E":-3.5,"F":2.8,"G":-0.4,"H":-3.2,"I":4.5,"K":-3.9,
    "L":3.8,"M":1.9,"N":-3.5,"P":-1.6,"Q":-3.5,"R":-4.5,"S":-0.8,"T":-0.7,"V":4.2,"W":-0.9,"Y":-1.3
}
EISENBERG = {
    "A":0.25,"R":-1.76,"N":-0.64,"D":-0.72,"C":0.04,"Q":-0.69,"E":-0.62,"G":0.16,"H":-0.40,"I":0.73,
    "L":0.53,"K":-1.10,"M":0.26,"F":0.61,"P":-0.07,"S":-0.26,"T":-0.18,"W":0.37,"Y":0.02,"V":0.54
}
PKA_POS = {"K":10.5,"R":12.5,"H":6.5}
PKA_NEG = {"D":3.9,"E":4.2,"C":8.3,"Y":10.1}
PKA_NTERM = 8.6
PKA_CTERM = 3.6
PHTARGET = 7.4

def mean_hydrophobicity(seq: str) -> float:
    vals = [KD.get(a, 0.0) for a in seq]
    return float(np.mean(vals)) if len(vals) else 0.0

def hydrophobic_moment(seq: str, angle_deg: float = 100.0) -> float:
    if not seq: return 0.0
    theta = math.radians(angle_deg)
    cx = cy = 0.0
    for i, a in enumerate(seq):
        h = EISENBERG.get(a, 0.0)
        cx += h * math.cos(theta * i)
        cy += h * math.sin(theta * i)
    mu = math.sqrt(cx*cx + cy*cy) / len(seq)
    return float(mu)

def net_charge_at_pH(seq: str, pH: float) -> float:
    if not seq: return 0.0
    q_nterm = 1.0 / (1.0 + 10.0**(pH - PKA_NTERM))
    q_cterm = -1.0 / (1.0 + 10.0**(PKA_CTERM - pH))
    pos = sum(seq.count(a) * (1.0 / (1.0 + 10.0**(pH - pka))) for a, pka in PKA_POS.items())
    neg = sum(seq.count(a) * (-1.0 / (1.0 + 10.0**(pka - pH))) for a, pka in PKA_NEG.items())
    return float(q_nterm + q_cterm + pos + neg)

def isoelectric_point(seq: str, tol: float = 1e-3) -> float:
    if not seq: return 0.0
    lo, hi = 0.0, 14.0
    q_lo, q_hi = net_charge_at_pH(seq, lo), net_charge_at_pH(seq, hi)
    if q_lo * q_hi > 0:
        return float(lo if abs(q_lo) < abs(q_hi) else hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        q_mid = net_charge_at_pH(seq, mid)
        if abs(q_mid) < tol:
            return float(mid)
        if q_mid * q_lo > 0:
            lo, q_lo = mid, q_mid
        else:
            hi, q_hi = mid, q_mid
    return float(0.5 * (lo + hi))

def compute_properties(seqs: List[str]) -> pd.DataFrame:
    rows = []
    for s in seqs:
        rows.append({
            "length": len(s),
            "hydrophobicity": mean_hydrophobicity(s),
            "hydrophobic_moment": hydrophobic_moment(s, 100.0),
            "charge": net_charge_at_pH(s, PHTARGET),
            "isoelectric_point": isoelectric_point(s)
        })
    return pd.DataFrame(rows)

# ----------------------------
# Tokenization & batching
# ----------------------------
def seq_to_indices(seq: str) -> List[int]:
    return [AA2IDX.get(a, 0) for a in seq if a in AA]

def series_to_indices(series: pd.Series) -> List[List[int]]:
    return [seq_to_indices(s) for s in series.astype(str)]

def pad_content(batch: List[List[int]], maxlen=MAX_CONTENT_LEN) -> np.ndarray:
    out = np.full((len(batch), maxlen), PAD_ID, dtype=np.int64)
    for i, seq in enumerate(batch):
        n = min(len(seq), maxlen)
        out[i, :n] = np.asarray(seq[:n], dtype=np.int64)
    return out

def make_in_tgt_pairs(content: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    N = len(content)
    x_in  = np.full((N, MAX_STEPS), PAD_ID, dtype=np.int64)
    x_tgt = np.full((N, MAX_STEPS), PAD_ID, dtype=np.int64)
    for i, seq in enumerate(content):
        seq = seq[:MAX_CONTENT_LEN]
        in_seq  = [BOS_ID] + seq
        tgt_seq = seq + [EOS_ID]
        L = min(len(in_seq), MAX_STEPS)
        x_in[i, :L]  = np.asarray(in_seq[:L], dtype=np.int64)
        x_tgt[i, :L] = np.asarray(tgt_seq[:L], dtype=np.int64)
    return x_in, x_tgt

def indices_to_seq_truncate_at_eos(ids: List[int]) -> str:
    s = []
    for t in ids:
        if t == EOS_ID or t == PAD_ID:
            break
        if 1 <= t <= 20:
            s.append(IDX2AA[int(t)])
    return "".join(s)

# ----------------------------
# Robust CSV loading
# ----------------------------
def _normalize_sequence_col(df: pd.DataFrame, want="Sequence") -> pd.DataFrame:
    """Rename any case-insensitive 'sequence' column to 'Sequence'."""
    if df is None or df.empty:
        return df
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    for c in cols:
        if c.lower() == "sequence" and c != want:
            df = df.rename(columns={c: want})
            break
    return df

def safe_read(path, required_any=None, encoding="utf-8-sig"):
    """
    Read CSV (BOM-safe), strip whitespace from column names.
    If required_any is provided, succeed if ANY case-insensitive name is present.
    """
    if not os.path.isfile(path):
        print(f"[WARN] Missing file: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding=encoding)
    df.columns = [str(c).strip() for c in df.columns]
    if required_any:
        have = {c.lower() for c in df.columns}
        ok = any(r.lower() in have for r in required_any)
        if not ok:
            raise ValueError(f"{path} missing column one of {required_any}")
    return df

def load_streams():
    # We are NOT using AMPdb anymore
    ampdb = pd.DataFrame(columns=["Sequence"])

    # Positives
    pos = safe_read(POS_CSV, required_any=["Sequence","sequence"])
    pos = _normalize_sequence_col(pos)
    if not pos.empty:
        pos["Sequence"] = pos["Sequence"].astype(str).str.strip().str.upper()
        pos = pos.loc[pos["Sequence"].str.len() <= MAX_CONTENT_LEN].copy()
    print(f"[POS] {len(pos) if not pos.empty else 0} rows")

    # Negatives (crop if needed to match length distribution)
    neg_raw = safe_read(NEG_CSV, required_any=["Sequence","sequence"])
    neg_raw = _normalize_sequence_col(neg_raw)
    neg = pd.DataFrame(columns=["Sequence"])

    if not neg_raw.empty:
        neg_raw["Sequence"] = neg_raw["Sequence"].astype(str).str.strip().str.upper()
        neg_leq25 = neg_raw.loc[neg_raw["Sequence"].str.len() <= MAX_CONTENT_LEN, ["Sequence"]].copy()

        if len(neg_leq25) > 0:
            neg = neg_leq25
        else:
            rng = random.Random(44)
            pos_lengths = pos["Sequence"].str.len().tolist() if not pos.empty else [MAX_CONTENT_LEN]
            if len(pos_lengths) == 0:
                pos_lengths = [MAX_CONTENT_LEN]

            neg_raw["len"] = neg_raw["Sequence"].str.len()
            crops = []
            for L in pos_lengths:
                pool = neg_raw.loc[neg_raw["len"] >= max(1, L)]
                if len(pool) == 0:
                    seq = rng.choice(neg_raw["Sequence"].tolist())
                    L_eff = min(len(seq), max(1, L))
                    start = rng.randrange(0, max(1, len(seq) - L_eff + 1))
                    crops.append(seq[start:start + L_eff])
                else:
                    seq = rng.choice(pool["Sequence"].tolist())
                    start = rng.randrange(0, max(1, len(seq) - L + 1))
                    crops.append(seq[start:start + L])

            neg = pd.DataFrame({"Sequence": crops})

    print(f"[NEG] {len(neg) if not neg.empty else 0} rows (≤{MAX_CONTENT_LEN} or cropped)")

    # MIC (optional, only as extra corpus sequences)
    mic = safe_read(MIC_CSV, required_any=["Sequence","sequence"])
    mic = _normalize_sequence_col(mic)
    if not mic.empty and "Sequence" in mic.columns:
        mic["Sequence"] = mic["Sequence"].astype(str).str.strip().str.upper()
        mic = mic.loc[mic["Sequence"].str.len() <= MAX_CONTENT_LEN].copy()
    print(f"[MIC] {len(mic) if not mic.empty else 0} rows")

    # UniProt unlabeled
    uni_tr = safe_read(UNI_TRAIN, required_any=["Sequence","sequence"])
    uni_tr = _normalize_sequence_col(uni_tr)
    if not uni_tr.empty:
        uni_tr["Sequence"] = uni_tr["Sequence"].astype(str).str.strip().str.upper()
        uni_tr = uni_tr.loc[uni_tr["Sequence"].str.len() <= MAX_CONTENT_LEN].copy()

    uni_va = safe_read(UNI_VAL, required_any=["Sequence","sequence"])
    uni_va = _normalize_sequence_col(uni_va)
    if not uni_va.empty:
        uni_va["Sequence"] = uni_va["Sequence"].astype(str).str.strip().str.upper()
        uni_va = uni_va.loc[uni_va["Sequence"].str.len() <= MAX_CONTENT_LEN].copy()

    # Build corpus
    train_seqs = []
    for df in [ampdb, pos, neg, mic, uni_tr]:
        if not df.empty and "Sequence" in df.columns:
            train_seqs += df["Sequence"].dropna().astype(str).tolist()

    val_seqs = (
        uni_va["Sequence"].dropna().astype(str).tolist()
        if not uni_va.empty
        else train_seqs[:2000]
    )

    train_idx = series_to_indices(pd.Series(train_seqs))
    val_idx   = series_to_indices(pd.Series(val_seqs))

    Xc_tr = pad_content(train_idx, MAX_CONTENT_LEN)
    Xc_va = pad_content(val_idx,   MAX_CONTENT_LEN)
    Xin_tr, Xtgt_tr = make_in_tgt_pairs(train_idx)
    Xin_va, Xtgt_va = make_in_tgt_pairs(val_idx)

    print(f"[DATA] Train sequences: {len(train_idx)} | Val sequences: {len(val_idx)}")
    return (Xc_tr, Xin_tr, Xtgt_tr), (Xc_va, Xin_va, Xtgt_va), ampdb  # ampdb stays empty

# ----------------------------
# Models
# ----------------------------
class AMPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb    = nn.Embedding(VOCAB_SIZE, 100, padding_idx=PAD_ID)
        self.bigru1 = nn.GRU(100, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.bigru2 = nn.GRU(2*HIDDEN_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.z_mu   = nn.Linear(2*HIDDEN_DIM, LATENT_DIM)
        self.z_lv   = nn.Linear(2*HIDDEN_DIM, LATENT_DIM)
    def forward(self, x_content_idx):
        e = self.emb(x_content_idx)           # (B,T,100)
        h,_ = self.bigru1(e)                  # (B,T,2H)
        h,_ = self.bigru2(h)                  # (B,T,2H)
        h = h[:, -1, :]                       # (B,2H)
        return self.z_mu(h), self.z_lv(h)

class ECDConditioner(nn.Module):
    def __init__(self, z_dim=LATENT_DIM, hidden=128, heads=4, use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        self.proj = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, z_dim)
        )
        if self.use_attn:
            self.q_seq = nn.Parameter(torch.randn(MAX_STEPS, z_dim))
            self.mha = nn.MultiheadAttention(z_dim, heads, batch_first=True)
    def forward(self, z):
        zc = z + self.proj(z)
        if self.use_attn:
            k = zc.unsqueeze(1); v = k
            q = self.q_seq.unsqueeze(0).expand(zc.size(0), -1, -1)
            attn, _ = self.mha(q, k, v)
            ctx = attn.mean(dim=1)
            zc = zc + ctx
        return zc

class ARDecoderECD(nn.Module):
    """
    Autoregressive decoder:
      - Training: consume x_in (BOS + content) to predict x_tgt (content + EOS).
      - Inference/RL: start from BOS, sample until EOS or MAX_STEPS.
    """
    def __init__(self):
        super().__init__()
        self.emb_tok = nn.Embedding(VOCAB_SIZE, 100, padding_idx=PAD_ID)
        self.fc_z    = nn.Linear(LATENT_DIM, HIDDEN_DIM)         # z → h0 for GRU
        self.gru     = nn.GRU(100, HIDDEN_DIM, batch_first=True) # token stream
        self.lstm    = nn.LSTM(HIDDEN_DIM, 100, batch_first=True)
        self.fc_out  = nn.Linear(100, VOCAB_SIZE)

    def forward(self, z, x_in):
        B, T = x_in.size()
        h0 = torch.tanh(self.fc_z(z)).unsqueeze(0)          # (1,B,128)
        e = self.emb_tok(x_in)                              # (B,T,100)
        y,_ = self.gru(e, h0)                               # (B,T,128)
        y,_ = self.lstm(y)                                  # (B,T,100)
        logits = self.fc_out(y)                             # (B,T,V)
        return logits

    def sample_ids_and_logp(self, zc, temp: float = 1.0, greedy: bool=False,
                            max_steps: int = MAX_STEPS) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive sampling with BOS; stops at EOS or max_steps.
        Returns:
          idx: (B, max_steps) sampled tokens (PAD for unused slots)
          logp_sum: (B,) sum of log-probs for emitted tokens (incl. EOS)
        """
        B = zc.size(0)
        h_gru  = torch.tanh(self.fc_z(zc)).unsqueeze(0)          # (1,B,128)
        h_lstm = torch.zeros(1, B, 100, device=zc.device)        # (1,B,100)
        c_lstm = torch.zeros(1, B, 100, device=zc.device)        # (1,B,100)

        prev = torch.full((B, 1), BOS_ID, dtype=torch.long, device=zc.device)
        idx_all = torch.full((B, max_steps), PAD_ID, dtype=torch.long, device=zc.device)
        logp_sum = torch.zeros(B, device=zc.device)
        done = torch.zeros(B, dtype=torch.bool, device=zc.device)

        for t in range(max_steps):
            e = self.emb_tok(prev)                               # (B,1,100)
            y_gru, h_gru = self.gru(e, h_gru)                   # (B,1,128)
            y_lstm, (h_lstm, c_lstm) = self.lstm(y_gru, (h_lstm, c_lstm))  # (B,1,100)
            logits = self.fc_out(y_lstm.squeeze(1))              # (B,V)
            if temp != 1.0:
                logits = logits / temp

            if greedy:
                next_tok = torch.argmax(logits, dim=-1)
                logp_t = F.log_softmax(logits, dim=-1).gather(1, next_tok.unsqueeze(1)).squeeze(1)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                next_tok = dist.sample()
                logp_t = dist.log_prob(next_tok)

            idx_all[:, t] = next_tok
            logp_sum = logp_sum + torch.where(done, torch.zeros_like(logp_t), logp_t)

            just_eos = (next_tok == EOS_ID)
            done = done | just_eos
            if done.all():
                break
            prev = next_tok.unsqueeze(1)

        return idx_all, logp_sum

# ----------------------------
# VAE wrapper
# ----------------------------
class MasterVAE(nn.Module):
    def __init__(self, enc: AMPEncoder, dec: ARDecoderECD):
        super().__init__()
        self.enc = enc
        self.dec = dec
        self.kl_w = torch.tensor(KL_MIN, device=DEVICE)

    def reparam(self, mu, lv, eps):
        return mu + torch.exp(0.5*lv) * eps

    def step(self, x_content, x_in, x_tgt, noise):
        mu, lv = self.enc(x_content)
        z = self.reparam(mu, lv, noise)                      # (B,D)
        logits = self.dec(z, x_in)                           # (B,T,V)

        ce = F.cross_entropy(logits.permute(0,2,1), x_tgt, reduction="none")  # (B,T)
        mask = (x_tgt != PAD_ID).float()                     # include EOS, exclude PAD
        rcl = (ce * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        kl  = -0.5 * torch.sum(1 + lv - mu.pow(2) - torch.exp(lv), dim=1)

        return {
            "loss": (RCL_WEIGHT * rcl + self.kl_w * kl).mean(),
            "rcl": rcl.mean(), "kl": kl.mean(),
            "mu": mu, "lv": lv, "z": z
        }

# ----------------------------
# Optional scorers
# ----------------------------
class NoConvAMPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(SCORER_VOCAB_SIZE, 128, padding_idx=PAD_ID)
        self.lstm1 = nn.LSTM(128, 64, batch_first=True)
        self.pool  = nn.MaxPool1d(5)
        self.lstm2 = nn.LSTM(64, 100, batch_first=True)
        self.fc = nn.Linear(100, 1)
    def forward(self, x_idx):
        x,_ = self.lstm1(self.emb(x_idx))
        x = self.pool(x.transpose(1,2)).transpose(1,2)
        x,_ = self.lstm2(x)
        h = x[:, -1, :]
        return torch.sigmoid(self.fc(h))  # AMP probability (high)

class VeltriAMPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(SCORER_VOCAB_SIZE, 128, padding_idx=PAD_ID)
        self.conv = nn.Conv1d(128, 64, kernel_size=16, padding=8)
        self.pool = nn.MaxPool1d(5)
        self.lstm = nn.LSTM(64, 100, batch_first=True)
        self.fc   = nn.Linear(100, 1)
    def forward(self, x_idx):
        e = self.emb(x_idx)
        y = F.relu(self.conv(e.transpose(1,2))).transpose(1,2)
        y = self.pool(y.transpose(1,2)).transpose(1,2)
        y,_ = self.lstm(y)
        h = y[:, -1, :]
        return torch.sigmoid(self.fc(h))  # MIC prob (1 = LOW MIC good)

def load_flexible_state_dict(model: nn.Module, path: str):
    sd = torch.load(path, map_location=DEVICE)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    msd = model.state_dict()
    new_sd, matched = {}, 0
    for k, v in sd.items():
        nk = k
        if nk not in msd and nk.startswith("embedding.") and ("emb." + nk.split("embedding.",1)[1]) in msd:
            nk = "emb." + nk.split("embedding.",1)[1]
        if nk in msd and msd[nk].shape == v.shape:
            new_sd[nk] = v; matched += 1
    model.load_state_dict(new_sd, strict=False)
    print(f"[LOAD] matched {matched}/{len(sd)} tensors for {os.path.basename(path)}")

def load_optional_scorers():
    amp_scorer = None
    if os.path.isfile(AMP_CLS_PT):
        amp_scorer = NoConvAMPClassifier().to(DEVICE)
        load_flexible_state_dict(amp_scorer, AMP_CLS_PT)
        amp_scorer.eval()
        print(f"[OK] Loaded AMP classifier: {AMP_CLS_PT}")
    else:
        print("[INFO] AMP classifier not found; skipping AMP scoring.")

    mic_scorer = None
    if os.path.isfile(MIC_CLS_PT):
        mic_scorer = VeltriAMPClassifier().to(DEVICE)
        load_flexible_state_dict(mic_scorer, MIC_CLS_PT)
        mic_scorer.eval()
        print(f"[OK] Loaded MIC classifier: {MIC_CLS_PT}  (label=1 means LOW MIC)")
    else:
        print("[INFO] MIC classifier not found; skipping MIC scoring.")

    return amp_scorer, mic_scorer

# ----------------------------
# Dataloaders
# ----------------------------
def array_gen_triples(Xc, Xin, Xtgt, batch):
    N = len(Xc); idx = np.arange(N)
    while True:
        np.random.shuffle(idx)
        for i in range(0, N, batch):
            sel = idx[i:i+batch]
            yield (Xc[sel], Xin[sel], Xtgt[sel])

def make_loader_triples(Xc, Xin, Xtgt, batch=BATCH):
    g = array_gen_triples(Xc, Xin, Xtgt, batch)
    while True:
        xc, xi, xt = next(g)
        noise = np.random.normal(0, 1.0, size=(xc.shape[0], LATENT_DIM)).astype(np.float32)
        yield (torch.tensor(xc, device=DEVICE),
               torch.tensor(xi, device=DEVICE),
               torch.tensor(xt, device=DEVICE),
               torch.tensor(noise, device=DEVICE))

# ----------------------------
# Train VAE
# ----------------------------
def anneal(epoch, start, rate, cap, mode):
    return min(cap, start * math.exp(rate*epoch)) if mode == "up" else max(cap, start * math.exp(-rate*epoch))

def train_vae(train_pack, val_pack):
    Xc_tr, Xin_tr, Xtgt_tr = train_pack
    Xc_va, Xin_va, Xtgt_va = val_pack

    enc, dec = AMPEncoder().to(DEVICE), ARDecoderECD().to(DEVICE)
    vae = MasterVAE(enc, dec).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=LR)

    train_iter = make_loader_triples(Xc_tr, Xin_tr, Xtgt_tr, BATCH)
    val_iter   = make_loader_triples(Xc_va, Xin_va, Xtgt_va, BATCH)

    steps_per_epoch = max(1, len(Xc_tr)//BATCH)
    val_steps = max(1, len(Xc_va)//BATCH)
    best_val = float("inf")

    print("\n[TRAIN] Starting VAE training (autoregressive)…")
    for ep in range(1, EPOCHS+1):
        vae.kl_w = torch.tensor(anneal(ep, KL_MIN, KL_RATE, KL_MAX, "up"), device=DEVICE)
        enc.train(); dec.train()
        run = []
        for step in range(1, steps_per_epoch+1):
            x_content, x_in, x_tgt, noise = next(train_iter)
            opt.zero_grad()
            out = vae.step(x_content, x_in, x_tgt, noise)
            out["loss"].backward()
            nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()), 1.0)
            opt.step()
            run.append(float(out["loss"].item()))
            if step % PRINT_EVERY == 0 or step == steps_per_epoch:
                print(f"[ep {ep:02d} | {step:04d}/{steps_per_epoch}] "
                      f"loss={np.mean(run):.3f} RCL={out['rcl'].item():.3f} "
                      f"KL={out['kl'].item():.3f} KLw={vae.kl_w.item():.5f}")
                run = []

        enc.eval(); dec.eval()
        with torch.no_grad():
            vlosses=[]
            for _ in range(min(val_steps, 10)):
                x_content, x_in, x_tgt, noise = next(val_iter)
                o = vae.step(x_content, x_in, x_tgt, noise)
                vlosses.append(float(o["loss"].item()))
            v = float(np.mean(vlosses))
        print(f"→ epoch {ep:02d} val_loss={v:.3f} {'[SAVE]' if v<best_val else ''}")
        if v < best_val:
            best_val = v
            torch.save({
                "encoder": enc.state_dict(),
                "decoder": dec.state_dict(),
                "kl_w": float(vae.kl_w.detach().cpu()),
                "config": {
                    "MAX_CONTENT_LEN": MAX_CONTENT_LEN, "MAX_STEPS": MAX_STEPS,
                    "LATENT_DIM": LATENT_DIM, "HIDDEN_DIM": HIDDEN_DIM,
                    "VOCAB_SIZE": VOCAB_SIZE
                }
            }, CKPT_PATH)

    torch.save(enc.state_dict(), ENC_PATH)
    torch.save(dec.state_dict(), DEC_PATH)
    print(f"[SAVE] Encoder -> {ENC_PATH}")
    print(f"[SAVE] Decoder -> {DEC_PATH}")
    return enc, dec

def load_best_or_train(train_pack, val_pack):
    enc, dec = AMPEncoder().to(DEVICE), ARDecoderECD().to(DEVICE)
    if os.path.isfile(CKPT_PATH):
        print(f"[LOAD] Found checkpoint → {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        enc.load_state_dict(ckpt["encoder"])
        dec.load_state_dict(ckpt["decoder"])
        print("[LOAD] Encoder/Decoder weights restored (autoregressive).")
        return enc, dec
    else:
        print("[LOAD] No checkpoint found → training VAE.")
        return train_vae(train_pack, val_pack)

# ----------------------------
# RL fine-tune (REINFORCE)
# ----------------------------
def rl_finetune_decoder(dec: ARDecoderECD, ecd: ECDConditioner,
                        amp_scorer: nn.Module, mic_scorer: nn.Module,
                        steps=RL_STEPS, batch=RL_BATCH, lr=RL_LR,
                        w_amp=RL_W_AMP, w_mic=RL_W_MIC, temp: float = 1.0):
    if amp_scorer is None and mic_scorer is None:
        print("[RL] Skipping: no scorers available.")
        return

    dec.train()
    for p in dec.parameters():
        p.requires_grad_(True)
    params = list(dec.parameters())

    if ecd is not None:
        ecd.train()
        for p in ecd.parameters():
            p.requires_grad_(True)
        params += list(ecd.parameters())

    opt = torch.optim.Adam(params, lr=lr)
    baseline = torch.zeros(1, device=DEVICE)

    def pad_for_scorers(seqs: List[str]) -> np.ndarray:
        idx_list = [seq_to_indices(s) for s in seqs]
        return pad_content(idx_list, MAX_CONTENT_LEN)

    def score_with_model(model, seqs):
        if model is None or len(seqs) == 0:
            return np.zeros(len(seqs), dtype=np.float32)
        X = pad_for_scorers(seqs)
        probs = []
        with torch.inference_mode():
            for i in range(0, len(X), 1024):
                xb = torch.tensor(X[i:i+1024], device=DEVICE)
                p = model(xb).squeeze(1).detach().cpu().numpy()
                probs.append(p)
        return np.concatenate(probs, axis=0) if len(probs) else np.zeros(len(seqs), dtype=np.float32)

    for step in range(1, steps + 1):
        z = torch.randn(batch, LATENT_DIM, device=DEVICE)
        zc = ecd(z) if ecd is not None else z

        idx, logp = dec.sample_ids_and_logp(zc, temp=temp, greedy=False, max_steps=MAX_STEPS)
        seqs = [indices_to_seq_truncate_at_eos(row.tolist()) for row in idx.detach().cpu()]

        with torch.no_grad():
            amp_prob_np = score_with_model(amp_scorer, seqs)   # AMP prob (high good)
            mic_prob_np = score_with_model(mic_scorer, seqs)   # MIC prob (low MIC good)
        amp_prob = torch.from_numpy(amp_prob_np).to(DEVICE).float()
        mic_prob = torch.from_numpy(mic_prob_np).to(DEVICE).float()

        reward = w_amp * amp_prob
        if mic_scorer is not None:
            reward = reward + (w_mic * mic_prob if MIC_POSITIVE_IS_LOW else -w_mic * mic_prob)
        reward = reward.clamp(-1.0, 1.0)

        baseline = 0.95 * baseline + 0.05 * reward.mean()
        advantage = reward - baseline

        loss = -(advantage.detach() * logp).mean()

        if ENTROPY_W > 0.0:
            token_counts = (idx != PAD_ID).float().sum(dim=1).clamp_min(1.0)
            loss = loss - ENTROPY_W * (logp / token_counts).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % 50 == 0:
            print(f"[RL {step:04d}/{steps}] loss={loss.item():.4f}  "
                  f"R(mean)={reward.mean().item():.3f}  AMP={amp_prob.mean().item():.3f}  MIC(low)={mic_prob.mean().item():.3f}")

# ----------------------------
# Generation (variable-length via EOS)
# ----------------------------
@torch.inference_mode()
def generate_de_novo(dec: ARDecoderECD, n=5000, temp: float = 1.0):
    """
    Formerly 'generate_unconstrained'.
    Samples random latent vectors and decodes them to brand-new sequences.
    """
    dec.eval()
    seqs, zs = [], []
    batch = 256
    while len(seqs) < n:
        z = torch.randn(batch, LATENT_DIM, device=DEVICE)
        idx, _ = dec.sample_ids_and_logp(z, temp=temp, greedy=False, max_steps=MAX_STEPS)
        s_batch = [indices_to_seq_truncate_at_eos(row.tolist()) for row in idx.detach().cpu()]
        for s, zrow in zip(s_batch, z.detach().cpu().numpy()):
            if s:
                seqs.append(s); zs.append(zrow)
            if len(seqs) >= n:
                break
    return seqs, np.array(zs)

@torch.inference_mode()
def generate_analogs(enc: AMPEncoder, dec: ARDecoderECD, seed_seqs: List[str],
                     n_per_seed=10, sigma=0.6, temp: float = 1.0):
    enc.eval(); dec.eval()
    seeds_idx = pad_content(series_to_indices(pd.Series(seed_seqs)), MAX_CONTENT_LEN)
    mu, lv = enc(torch.tensor(seeds_idx, device=DEVICE))
    seqs, src, zs = [], [], []
    for i in range(len(seed_seqs)):
        for _ in range(n_per_seed):
            z = mu[i:i+1] + sigma*torch.randn(1, LATENT_DIM, device=DEVICE)
            idx, _ = dec.sample_ids_and_logp(z, temp=temp, greedy=False, max_steps=MAX_STEPS)
            s = indices_to_seq_truncate_at_eos(idx[0].detach().cpu().tolist())
            if s:
                seqs.append(s); src.append(seed_seqs[i]); zs.append(z.squeeze(0).cpu().numpy())
    return seqs, src, np.array(zs)

@torch.inference_mode()
def score_sequences(scorer, seqs):
    if scorer is None or len(seqs) == 0:
        return np.full(len(seqs), np.nan)
    idx = pad_content([seq_to_indices(s) for s in seqs], MAX_CONTENT_LEN)
    probs = []
    for i in range(0, len(idx), 1024):
        xb = torch.tensor(idx[i:i+1024], device=DEVICE)
        p = scorer(xb).squeeze(1).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0) if len(probs) else np.full(len(seqs), np.nan)

def save_fasta(path, names, seqs, extra_headers=None):
    with open(path, "w") as f:
        for i,(n,s) in enumerate(zip(names, seqs)):
            header = n
            if extra_headers is not None:
                header += " " + extra_headers[i]
            f.write(f">{header}\n{s}\n")

# ----------------------------
# Main
# ----------------------------
def main():
    train_pack, val_pack, ampdb_df = load_streams()  # ampdb_df unused/empty

    # Load best or train
    enc, dec = load_best_or_train(train_pack, val_pack)

    # Optional scorers
    amp_scorer, mic_scorer = load_optional_scorers()

    # RL fine-tune
    ecd = ECDConditioner().to(DEVICE)
    if DO_RL and (amp_scorer is not None or mic_scorer is not None):
        print("\n[RL] Fine-tuning decoder with AMP/MIC rewards …")
        rl_finetune_decoder(dec, ecd, amp_scorer, mic_scorer, temp=1.0)

    # ---------- De novo generation (formerly 'Unconstrained') ----------
    print("\n[GEN] De novo …")
    dn_seqs, _ = generate_de_novo(dec, n=5000, temp=1.0)

    # scores
    dn_amp = score_sequences(amp_scorer, dn_seqs)
    dn_mic = score_sequences(mic_scorer, dn_seqs)  # prob of LOW MIC

    # properties
    dn_props = compute_properties(dn_seqs)

    df_dn = pd.DataFrame({
        "name": [f"denovo_{i:05d}" for i, _ in enumerate(dn_seqs)],
        "sequence": dn_seqs,
        "amp_prob": dn_amp,              # probability of HIGH AMP
        "mic_prob_low": dn_mic,          # probability of LOW MIC (good)
    })
    df_dn = pd.concat([df_dn, dn_props], axis=1)

    dn_csv = os.path.join(OUT_DIR, "generated_denovo_ecd_ar.csv")
    dn_fa  = os.path.join(OUT_DIR, "generated_denovo_ecd_ar.fasta")
    df_dn.to_csv(dn_csv, index=False)

    headers_dn = [
        f"amp={a:.3f} mic_low={m:.3f} len={l}"
        for a,m,l in zip(df_dn["amp_prob"], df_dn["mic_prob_low"], df_dn["length"])
    ]
    save_fasta(dn_fa, df_dn["name"].tolist(), df_dn["sequence"].tolist(), headers_dn)

    print(f"[SAVE] De novo -> {dn_csv} | {dn_fa}")
    print(df_dn.head(5).to_string(index=False))

    # ---------- Analog generation (seed from training corpus only) ----------
    print("\n[GEN] Analogs …")
    Xc_tr, _, _ = train_pack
    seeds = []
    for i in range(min(200, len(Xc_tr))):
        s = indices_to_seq_truncate_at_eos(Xc_tr[i].tolist())
        if s:
            seeds.append(s)
    seeds = seeds[:50]

    an_seqs, an_src, _ = generate_analogs(enc, dec, seeds, n_per_seed=100, sigma=0.6, temp=1.0)
    an_amp = score_sequences(amp_scorer, an_seqs)
    an_mic = score_sequences(mic_scorer, an_seqs)
    an_props = compute_properties(an_seqs)

    df_an = pd.DataFrame({
        "name": [f"analog_{i:05d}" for i, _ in enumerate(an_seqs)],
        "seed": an_src,
        "sequence": an_seqs,
        "amp_prob": an_amp,
        "mic_prob_low": an_mic,
    })
    df_an = pd.concat([df_an, an_props], axis=1)

    an_csv = os.path.join(OUT_DIR, "generated_analogs_ecd_ar.csv")
    an_fa  = os.path.join(OUT_DIR, "generated_analogs_ecd_ar.fasta")
    df_an.to_csv(an_csv, index=False)

    headers_an = [
        f"seed={seed} amp={a:.3f} mic_low={m:.3f} len={l}"
        for seed,a,m,l in zip(df_an["seed"], df_an["amp_prob"], df_an["mic_prob_low"], df_an["length"])
    ]
    save_fasta(an_fa, df_an["name"].tolist(), df_an["sequence"].tolist(), headers_an)

    print(f"[SAVE] Analogs -> {an_csv} | {an_fa}")
    print(df_an.head(5).to_string(index=False))

    print("\n[DONE] Models saved in 'models/'. Generated peptides saved in 'Result_1ststage/'.")

if __name__ == "__main__":
    main()
