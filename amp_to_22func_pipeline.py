# amp_to_22func_pipeline.py
# -*- coding: utf-8 -*-

import os, sys, re, json, glob, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AutoTokenizer, EsmModel

# ========================== I/O CONFIG ==========================
# Stage-1 generation outputs (inputs to this pipeline)
UN_PATH = "Result_1ststage/generated_denovo_ecd_ar.csv"
AN_PATH = "Result_1ststage/generated_analogs_ecd_ar.csv"

# Directories for saving pipeline outputs
OUT_DIR_STAGE2 = "Result_after_2ndstage"   # after Stage-2 AMP ensemble
OUT_DIR_STAGE3 = "Final_Result"            # after Stage-3 functional profiling
os.makedirs(OUT_DIR_STAGE2, exist_ok=True)
os.makedirs(OUT_DIR_STAGE3, exist_ok=True)

# Stage-1 (AMP ensemble) model weights (used in Stage 2)
MODEL_1ST = dict(
    token_protbert="Model/Model_2ndstage/best_Bert_cnn_bilstm_attn_fold6.pth",
    token_esm2="Model/Model_2ndstage/best_esm2_cnn_bilstm_attn_fold6.pth",
    ft_protbert="Model/Model_2ndstage/cv_protbert_cnn_bilstm_attn_FT_fold6.pth",
    ft_esm2="Model/Model_2ndstage/cv_esm2_cnn_bilstm_attn_FT_HP_fold6.pth",
)

# Stage-2 (22-function) artifacts root (must contain specialists/ and stacker_from_saved/)
MODEL_2ND_ROOT = "Model/Model_3rdstage"
SPECIALISTS_FOLD = "fold_1"
N_STACK_FOLDS = 10
SEQ_COL_CANDIDATES = ["sequence", "Sequence", "seq", "Seq", "AASequence"]

# Stage-2 decision rule: "hard" = majority vote; "soft" = avg >= THRESH; "both" = both true
DECISION_MODE = "soft"
SOFT_THRESHOLD = 0.50

# Stage-2 batching
BATCH_SIZE = 16
MAX_SEQ_TOKENS = 512

# Stage-3 features / thresholds (must match training)
MAX_LEN_FEATS = 200
BASE_HIDDEN = 512
BASE_HIDDEN2 = 256
FUNC_THRESHOLD = 0.50    # 0/1 cutoff for each functional attribute

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================================================
# ---------------- Stage-2: AMP Ensemble ------------------------
def pick_seq_col(df):
    for c in SEQ_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No sequence column found; expected one of: {SEQ_COL_CANDIDATES}")

def protbert_tokenize(tok, seqs):
    # ProtBERT expects space-separated amino acids
    spaced = [" ".join(list(s)) for s in seqs]
    return tok(spaced, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_TOKENS)

def esm_tokenize(tok, seqs):
    return tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_TOKENS)

def pad_to(x, L):
    if x.size(1) >= L:
        return x[:, :L, :]
    out = torch.zeros(x.size(0), L, x.size(2), device=x.device, dtype=x.dtype)
    out[:, :x.size(1), :] = x
    return out

class CNN_BiLSTM_Attention(nn.Module):
    # token-level heads that take embeddings directly
    def __init__(self, embed_dim=1024, cnn_out=256, lstm_hidden=128, num_classes=2):
        super().__init__()
        self.conv = nn.Conv1d(embed_dim, cnn_out, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bilstm = nn.LSTM(cnn_out, lstm_hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * lstm_hidden, 1)
        self.fc = nn.Linear(2 * lstm_hidden, num_classes)
    def forward(self, x_emb):
        # x_emb: (B, T, E)
        x = x_emb.transpose(1, 2)                           # (B, E, T)
        x = self.relu(self.conv(x))                         # (B, C, T)
        x = self.dropout(x)
        x = x.transpose(1, 2)                               # (B, T, C)
        h_lstm, _ = self.bilstm(x)                          # (B, T, 2*H)
        attn = torch.softmax(self.attn(h_lstm), dim=1)      # (B, T, 1)
        ctx = (h_lstm * attn).sum(dim=1)                    # (B, 2*H)
        return self.fc(ctx)                                 # (B, num_classes)

class CNN_BiLSTM_Attn(nn.Module):
    # end-to-end heads that include the encoder
    def __init__(self, encoder, embed_dim, cnn_out=256, lstm_hidden=128, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.conv = nn.Conv1d(embed_dim, cnn_out, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bilstm = nn.LSTM(cnn_out, lstm_hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * lstm_hidden, 1)
        self.fc = nn.Linear(2 * lstm_hidden, num_classes)
    def forward(self, tokens):
        # tokens = tokenizer output dict
        x = self.encoder(**tokens).last_hidden_state         # (B, T, E)
        # drop [CLS]/special tokens if present
        x = x[:, 1:-1, :] if x.size(1) > 2 else x
        x = x.transpose(1, 2)                                # (B, E, T)
        x = self.relu(self.conv(x))                          # (B, C, T)
        x = self.dropout(x)
        x = x.transpose(1, 2)                                # (B, T, C)
        h_lstm, _ = self.bilstm(x)                           # (B, T, 2*H)
        attn = torch.softmax(self.attn(h_lstm), dim=1)       # (B, T, 1)
        ctx = (h_lstm * attn).sum(dim=1)                     # (B, 2*H)
        return self.fc(ctx)                                  # (B, num_classes)

def build_ensemble_stage2():
    models = []

    # Load tokenizers/encoders
    protbert_tok = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    esm_tok      = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    protbert_encoder = BertModel.from_pretrained("Rostlab/prot_bert").to(device)
    esm_encoder      = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(device)

    # 1. token_protbert head (frozen encoder, external embeddings)
    if os.path.exists(MODEL_1ST["token_protbert"]):
        m1 = CNN_BiLSTM_Attention(embed_dim=1024).to(device)
        m1.load_state_dict(torch.load(MODEL_1ST["token_protbert"], map_location=device))
        m1.eval()
        models.append(("token_protbert", m1))
    else:
        print(f"⚠️ Missing {MODEL_1ST['token_protbert']} (skipping)")

    # 2. token_esm2 head
    if os.path.exists(MODEL_1ST["token_esm2"]):
        m2 = CNN_BiLSTM_Attention(embed_dim=480).to(device)
        m2.load_state_dict(torch.load(MODEL_1ST["token_esm2"], map_location=device))
        m2.eval()
        models.append(("token_esm2", m2))
    else:
        print(f"⚠️ Missing {MODEL_1ST['token_esm2']} (skipping)")

    # 3. ft_protbert head (finetuned ProtBERT+CNN+BiLSTM+Attn jointly)
    if os.path.exists(MODEL_1ST["ft_protbert"]):
        ft1 = CNN_BiLSTM_Attn(protbert_encoder, embed_dim=1024).to(device)
        ft1.load_state_dict(torch.load(MODEL_1ST["ft_protbert"], map_location=device))
        ft1.eval()
        models.append(("ft_protbert", ft1))
    else:
        print(f"⚠️ Missing {MODEL_1ST['ft_protbert']} (skipping)")

    # 4. ft_esm2 head
    if os.path.exists(MODEL_1ST["ft_esm2"]):
        ft2 = CNN_BiLSTM_Attn(esm_encoder, embed_dim=480).to(device)
        ft2.load_state_dict(torch.load(MODEL_1ST["ft_esm2"], map_location=device))
        ft2.eval()
        models.append(("ft_esm2", ft2))
    else:
        print(f"⚠️ Missing {MODEL_1ST['ft_esm2']} (skipping)")

    if len(models) == 0:
        raise RuntimeError("Stage-2: No models found in Model_2ndstage/*")

    # Freeze all params
    for _, mdl in models:
        for p in mdl.parameters():
            p.requires_grad_(False)

    return models, protbert_tok, esm_tok, protbert_encoder, esm_encoder

@torch.no_grad()
def score_batch_stage2(batch_seqs, models, pb_tok, esm_tok, pb_enc, esm_enc):
    # Tokenize for ProtBERT and ESM2
    pb_tokens = protbert_tokenize(pb_tok, batch_seqs).to(device)
    esm_tokens = esm_tokenize(esm_tok, batch_seqs).to(device)

    # Get hidden states from encoders
    pb_hidden = pb_enc(**pb_tokens).last_hidden_state  # (B, T, 1024)
    esm_hidden = esm_enc(**esm_tokens).last_hidden_state  # (B, T, 480)

    # Drop special tokens ([CLS],[SEP]) where appropriate
    if pb_hidden.size(1) > 2:
        pb_hidden = pb_hidden[:, 1:-1, :]
    if esm_hidden.size(1) > 1:
        esm_hidden = esm_hidden[:, 1:, :]

    # Pad/trim to max length for CNN/LSTM heads
    pb_hidden = pad_to(pb_hidden, MAX_SEQ_TOKENS)
    esm_hidden = pad_to(esm_hidden, MAX_SEQ_TOKENS)

    probs = []
    for name, mdl in models:
        if name == "token_protbert":
            p = torch.softmax(mdl(pb_hidden), dim=1)[:, 1]
        elif name == "token_esm2":
            p = torch.softmax(mdl(esm_hidden), dim=1)[:, 1]
        elif name == "ft_protbert":
            p = torch.softmax(mdl(pb_tokens), dim=1)[:, 1]
        elif name == "ft_esm2":
            p = torch.softmax(mdl(esm_tokens), dim=1)[:, 1]
        else:
            continue
        probs.append(p.unsqueeze(0))

    # Concatenate model-wise probs -> ensemble
    P = torch.cat(probs, dim=0)             # (num_models, B)
    avg = P.mean(dim=0)                     # soft score
    hard = (P > 0.5).int().sum(dim=0) >= (P.size(0)//2 + 1)  # majority vote

    return avg.cpu().numpy(), hard.int().cpu().numpy()

def stage2_tag_and_save(path, models, pb_tok, esm_tok, pb_enc, esm_enc):
    """
    Stage-2:
    - Read input CSV
    - Score each sequence with ensemble
    - Add columns:
        ensemble_softscore, ensemble_amp
    - Save tagged CSV into Result_after_2ndstage/
    - Return (tagged_path, seq_col)
    """
    df = pd.read_csv(path)
    seq_col = pick_seq_col(df)
    seqs = df[seq_col].astype(str).str.strip().tolist()

    soft_scores = np.zeros(len(seqs), dtype=np.float32)
    hard_votes  = np.zeros(len(seqs), dtype=np.int32)

    for i in tqdm(range(0, len(seqs), BATCH_SIZE), desc=f"[Stage-2] {os.path.basename(path)}"):
        batch = seqs[i:i+BATCH_SIZE]
        avg, hard = score_batch_stage2(batch, models, pb_tok, esm_tok, pb_enc, esm_enc)
        soft_scores[i:i+len(batch)] = avg
        hard_votes[i:i+len(batch)]  = hard

    # Decision logic
    if DECISION_MODE == "hard":
        ensemble_amp = hard_votes
    elif DECISION_MODE == "soft":
        ensemble_amp = (soft_scores >= SOFT_THRESHOLD).astype(int)
    elif DECISION_MODE == "both":
        ensemble_amp = ((soft_scores >= SOFT_THRESHOLD) & (hard_votes == 1)).astype(int)
    else:
        raise ValueError("DECISION_MODE must be 'hard', 'soft', or 'both'")

    # Attach columns
    df["ensemble_softscore"] = soft_scores
    df["ensemble_amp"] = ensemble_amp.astype(int)

    # Build save path for Stage-2 output
    base_name = os.path.splitext(os.path.basename(path))[0]         # e.g. "generated_denovo_ecd_ar"
    tagged_path = os.path.join(OUT_DIR_STAGE2, f"{base_name}_tagged.csv")

    df.to_csv(tagged_path, index=False)
    print(f"→ Wrote {tagged_path} with 'ensemble_amp'")

    before_rows = len(df)
    after_rows = int(df["ensemble_amp"].sum())
    print(f"   pass: {after_rows}/{before_rows} ({after_rows / max(1,before_rows) * 100:.2f}%)")

    return tagged_path, seq_col, before_rows, after_rows

# ===============================================================
# ---------------- Stage-3: 22-function pred --------------------
def import_data_feature():
    """Import your local data_feature.py that builds AAI/BLOSUM62/PAAC/OneHot."""
    here = os.path.abspath(os.path.dirname(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import data_feature as DF
    return DF

@torch.no_grad()
def seq_embeddings(seqs, max_len):
    DF = import_data_feature()
    aai  = DF.AAI_embedding(seqs, max_len=max_len).float()
    bl62 = DF.BLOSUM62_embedding(seqs, max_len=max_len).float()
    paac = DF.PAAC_embedding(seqs, max_len=max_len).float()
    oneh = DF.onehot_embedding(seqs, max_len=max_len).float()
    return aai, bl62, paac, oneh

def pool_mean_std_max(x: torch.Tensor):
    # Aggregate per-position features -> fixed-length vector
    return torch.cat([x.mean(1), x.std(1, unbiased=False), x.amax(1)], dim=1)

def build_feature_matrix(seqs, max_len):
    aai, bl62, paac, oneh = seq_embeddings(seqs, max_len)
    feats = [pool_mean_std_max(aai), pool_mean_std_max(bl62),
             pool_mean_std_max(paac), pool_mean_std_max(oneh)]
    return torch.cat(feats, 1).cpu().numpy().astype(np.float32)

def _norm(s: str) -> str:
    return re.sub(r'[^0-9a-z]+', '', s.lower())

class MLPBinary(nn.Module):
    def __init__(self, in_dim, hidden1=BASE_HIDDEN, hidden2=BASE_HIDDEN2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.BatchNorm1d(hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

class MLPStacker(nn.Module):
    def __init__(self, in_dim=22, hidden=64, dropout=0.1, out_dim=22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

def _safe_load_state_dict(model: nn.Module, path: str, device: torch.device, strict: bool = False):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, nn.Module):
        sd = sd.state_dict()
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        print(f"⚠️ Missing keys in {os.path.basename(path)}: {missing}")
    if unexpected:
        print(f"⚠️ Unexpected keys in {os.path.basename(path)}: {unexpected}")
    model.eval()
    return model

def _sanitize_labels(labels):
    return [re.sub(r'[^0-9A-Za-z_]+', '_', str(L)).strip('_') for L in labels]

def _build_filename_map(spec_dir):
    file_map = {}
    for p in glob.glob(os.path.join(spec_dir, "*.pth")):
        base = os.path.splitext(os.path.basename(p))[0]
        file_map[_norm(base)] = p
    return file_map

def _load_label_order(model_root, specialists_fold):
    # Prefer explicit label list file
    for cand in [
        os.path.join(model_root, "labels.json"),
        os.path.join(model_root, "label_cols.json"),
        os.path.join(model_root, "specialists", "labels.json"),
    ]:
        if os.path.exists(cand):
            with open(cand, "r") as f:
                data = json.load(f)
            labels = data.get("labels", data)
            labels = list(labels)
            return _sanitize_labels(labels)

    # Fallback from filenames in specialists dir
    spec_dir = os.path.join(model_root, "specialists", specialists_fold)
    pths = sorted(glob.glob(os.path.join(spec_dir, "*.pth")))
    if not pths:
        raise FileNotFoundError(f"No specialist .pth files found in {spec_dir}")
    labels = [os.path.splitext(os.path.basename(p))[0] for p in pths]
    print("ℹ️ Using specialists filenames as label order:")
    for L in labels:
        print(" -", L)
    return _sanitize_labels(labels)

def _resolve_weight_path_for_label(spec_dir, label_display, filename_map):
    bases_to_try = [
        label_display,
        label_display.replace('_', '-'),
        label_display.replace('_', ' ')
    ]
    for b in bases_to_try:
        cand = os.path.join(spec_dir, f"{b}.pth")
        if os.path.exists(cand):
            return cand
    norm = _norm(label_display)
    if norm in filename_map:
        return filename_map[norm]
    avail = sorted([os.path.basename(p) for p in glob.glob(os.path.join(spec_dir, "*.pth"))])
    raise FileNotFoundError(
        f"Missing specialist weight for label '{label_display}'. "
        f"Tried {bases_to_try}. Available: {avail}"
    )

def load_scaler(model_root, specialists_fold):
    p = os.path.join(model_root, "specialists", specialists_fold, "scaler.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Scaler not found: {p}")
    with open(p, "rb") as f:
        return pickle.load(f)

def load_specialists(model_root, specialists_fold, in_dim):
    labels = _load_label_order(model_root, specialists_fold)
    spec_dir = os.path.join(model_root, "specialists", specialists_fold)
    filename_map = _build_filename_map(spec_dir)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for lbl in labels:
        mp = _resolve_weight_path_for_label(spec_dir, lbl, filename_map)
        m = MLPBinary(in_dim=in_dim, hidden1=BASE_HIDDEN, hidden2=BASE_HIDDEN2).to(dev)
        _safe_load_state_dict(m, mp, dev, strict=False)
        models.append(m)
    return labels, models, dev

def load_stackers(model_root, n_folds, in_dim):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for k in range(1, n_folds+1):
        mp = os.path.join(model_root, "stacker_from_saved", f"fold_{k}", "meta.pth")
        if os.path.exists(mp):
            m = MLPStacker(in_dim=in_dim, hidden=64, dropout=0.1, out_dim=in_dim).to(dev)
            _safe_load_state_dict(m, mp, dev, strict=False)
            models.append(m)
        else:
            print(f"⚠️ Missing stacker fold {k}: {mp}")
    return models, dev

@torch.no_grad()
def predict_22func_for_seqs(seqs):
    """
    Returns:
        mean_probs: (N, C) probs in [0,1]
        labels: list[str] of C function names
    """
    if len(seqs) == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    # Step 1: feature extraction
    X_raw = build_feature_matrix(seqs, MAX_LEN_FEATS)

    # Step 2: scale features
    scaler = load_scaler(MODEL_2ND_ROOT, SPECIALISTS_FOLD)
    X = scaler.transform(X_raw).astype(np.float32)

    # Step 3: specialist per-function binary predictors
    labels, specialists, dev = load_specialists(MODEL_2ND_ROOT, SPECIALISTS_FOLD, in_dim=X.shape[1])
    xb = torch.from_numpy(X).to(dev)
    C = len(labels)

    probs_spec = np.zeros((len(seqs), C), dtype=np.float32)
    for j, m in enumerate(specialists):
        probs_spec[:, j] = torch.sigmoid(m(xb)).cpu().numpy()

    # Step 4: stackers (meta-models to capture co-occurrence)
    stackers, dev2 = load_stackers(MODEL_2ND_ROOT, N_STACK_FOLDS, in_dim=C)
    if len(stackers) == 0:
        mean_probs = probs_spec
    else:
        inp = torch.from_numpy(probs_spec).to(dev2)
        preds = [torch.sigmoid(m(inp)).cpu().numpy() for m in stackers]
        mean_probs = np.mean(preds, axis=0)

    return mean_probs, labels

def stage3_predict_on_amp_pass(tagged_path, seq_col):
    """
    Stage-3:
    - Load the Stage-2 tagged CSV from Result_after_2ndstage/
    - Filter to ensemble_amp == 1
    - Predict 22 functions
    - Save AMP+ rows with predictions to Final_Result/
    """
    df = pd.read_csv(tagged_path)
    if "ensemble_amp" not in df.columns:
        raise ValueError(f"{tagged_path} missing 'ensemble_amp'. Run Stage-2 first.")

    df_pass = df[df["ensemble_amp"] == 1].copy()
    if df_pass.empty:
        print(f"[Stage-3] No AMP-positive rows in {os.path.basename(tagged_path)}; skipping.")
        return None

    seqs = df_pass[seq_col].astype(str).str.strip().tolist()
    print(f"[Stage-3] Predicting 22 attributes for {len(seqs)} AMP-positive sequences from {os.path.basename(tagged_path)}")
    mean_probs, labels = predict_22func_for_seqs(seqs)

    # append per-function probabilities + binary calls
    for j, lbl in enumerate(labels):
        df_pass[f"{lbl}_prob"] = mean_probs[:, j]
        df_pass[lbl] = (mean_probs[:, j] >= FUNC_THRESHOLD).astype(int)

    # build save path for Stage-3 output
    base_name = os.path.splitext(os.path.basename(tagged_path))[0]   # e.g. "generated_denovo_ecd_ar_tagged"
    out_path = os.path.join(OUT_DIR_STAGE3, f"{base_name}_amp_22func_pred.csv")

    df_pass.to_csv(out_path, index=False)
    print(f"→ Wrote {out_path} (AMP-positive rows with 22-function predictions)")
    return out_path

# ===============================================================
# ---------------------------- Main -----------------------------
def main():
    # Stage-2: build AMP ensemble
    models, pb_tok, esm_tok, pb_enc, esm_enc = build_ensemble_stage2()

    # We'll process both de novo and analog sources
    # "source_name" is just for naming the summary .csv
    for source_name, in_path in [
        ("denovo",  UN_PATH),
        ("analogs", AN_PATH),
    ]:
        if not os.path.exists(in_path):
            print(f"⚠️ Missing input: {in_path} (skipping)")
            continue

        # Stage-2 scoring and save
        tagged_path, seq_col, before_rows, after_rows = stage2_tag_and_save(
            in_path, models, pb_tok, esm_tok, pb_enc, esm_enc
        )

        # Save summary counts for Stage-2 into Result_after_2ndstage
        summary = pd.DataFrame([{
            "source": source_name,
            "before_rows": before_rows,
            "after_rows_amp1": after_rows,
            "pass_rate_pct": round(after_rows / max(1, before_rows) * 100, 2)
        }])
        counts_path = os.path.join(OUT_DIR_STAGE2, f"{source_name}_ensemble_counts.csv")
        summary.to_csv(counts_path, index=False)
        print(f"→ Saved counts: {counts_path}")

        # Stage-3: functional profiling for AMP-positive only, saved to Final_Result
        stage3_predict_on_amp_pass(tagged_path, seq_col)

if __name__ == "__main__":
    main()
