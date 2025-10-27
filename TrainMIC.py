# mic_classifier_pytorch_spyder.py
# HydrAMP-style MIC classifier in PyTorch w/ paper-exact labeling.
# - Length filter: sequence length ≤ 25
# - Label rule: positive if log10(MIC) ≤ 1.5 (i.e., MIC ≤ 10^1.5)
# - Model: Emb(21,128) -> Conv1d(64,k=16) -> MaxPool(5) -> LSTM(100) -> Dense(1,sigmoid)
# - CV: 10-fold StratifiedKFold, oversample positives ×5 inside each fold
# - Early stopping: patience=7 on **best validation F1** (AUC printed)
# - Best saving: write a file on each F1 improvement (like Keras logger); restore best before eval
# - Final training: fit on full training split (w/ oversampling), validate on MIC test split, save best

import os, random
from typing import List, Tuple
import numpy as np
import pandas as pd

# ========= Config =========
SEED = 36
MIN_LENGTH = 0
MAX_LENGTH = 25
TEST_SIZE  = 0.10
BATCH_SIZE = 64
EPOCHS     = 32
PATIENCE   = 7
LR         = 1e-3
THRESH_BIN = 0.5
MIC_LOG_CUTOFF = 1.5  # paper rule

DATA_MIC_PATH   = "Dataset/1ststage/mic_data.csv"               # 3rd column must be log10(MIC)
DATA_POS_UNLAB  = "Dataset/1ststage/unlabelled_positive.csv"    # Name, sequence  (≤25 aa)
DATA_NEG_UNLAB  = "Dataset/1ststage/unlabelled_negative.csv"    # Name, sequence  (can be long)
SAVE_DIR = "models"; os.makedirs(SAVE_DIR, exist_ok=True)

# ========= Reproducibility =========
np.random.seed(SEED)
random.seed(SEED)
import torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ========= Tokenization / padding (1..20; PAD=0) =========
ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_ID = {aa: i for i, aa in enumerate(ALPHABET, start=1)}  # 1..20; 0=PAD

def encode_seq(s: str) -> List[int]:
    s = str(s).strip().upper()
    return [AA_TO_ID.get(ch, 0) for ch in s]

def pad_sequences(idxs: List[List[int]], maxlen: int) -> np.ndarray:
    out = np.zeros((len(idxs), maxlen), dtype=np.int64)
    for i, arr in enumerate(idxs):
        arr = arr[:maxlen]
        out[i, :len(arr)] = np.asarray(arr, dtype=np.int64)
    return out

# ========= Helpers =========
def detect_sequence_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).lower() == "sequence":
            return c
    for c in df.columns:
        try:
            if df[c].astype(str).map(lambda v: isinstance(v, str)).mean() > 0.5:
                return c
        except Exception:
            pass
    return df.columns[0]

def normalize_sequence_col(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.assign(sequence=[])
    seq_col = detect_sequence_col(df)
    if seq_col != "sequence":
        df = df.rename(columns={seq_col: "sequence"})
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    return df

def filter_by_length(df: pd.DataFrame, min_len: int, max_len: int) -> pd.DataFrame:
    m = df["sequence"].str.len().between(min_len, max_len, inclusive="both")
    return df.loc[m].copy()

# ========= Build MIC dataset (paper-exact): len ≤25, label from log10(MIC) ≤ 1.5 =========
mic_raw = pd.read_csv(DATA_MIC_PATH, encoding="utf-8-sig")
if mic_raw.shape[1] < 3:
    raise ValueError("mic_data.csv must have ≥3 columns; the 3rd column must be log10(MIC).")

seq_col   = detect_sequence_col(mic_raw)   # usually 'sequence'
value_col = mic_raw.columns[2]             # 3rd column is log10(MIC)

df = mic_raw.copy()
df[seq_col] = df[seq_col].astype(str).str.strip().str.upper()

# length ≤ 25 and label from stored log10(MIC)
mask = df[seq_col].str.len().between(MIN_LENGTH, MAX_LENGTH, inclusive="both")
sub  = df.loc[mask, [seq_col, value_col]].copy().rename(columns={seq_col: "sequence", value_col: "log10_mic"})

sub["log10_mic"] = pd.to_numeric(sub["log10_mic"], errors="coerce")
sub = sub.dropna(subset=["log10_mic"])
sub["label"] = (sub["log10_mic"] <= MIC_LOG_CUTOFF).astype(int)

mic_idx = [encode_seq(s) for s in sub["sequence"].tolist()]
mic_x   = pad_sequences(mic_idx, MAX_LENGTH)
mic_y   = sub["label"].astype(np.int64).values

print(f"Sequence column: '{seq_col}' | log10(MIC) column (3rd col): '{value_col}'")
print(f"Length ≤ {MAX_LENGTH}: {len(sub)}")
print(f"Label rule (paper): label=1 if log10(MIC) ≤ {MIC_LOG_CUTOFF} (MIC ≤ 10^{MIC_LOG_CUTOFF})")
print(f"Positives (label=1): {(mic_y==1).sum()} | Negatives (label=0): {(mic_y==0).sum()}")

# ========= Unlabelled data (positives ≤25; negatives cropped to match pos length distribution) =========
def load_unlabelled_xy(pos_csv: str, neg_csv: str, min_len: int, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (os.path.exists(pos_csv) and os.path.exists(neg_csv)):
        return np.zeros((0, max_len), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    pos_df = normalize_sequence_col(pd.read_csv(pos_csv, encoding="utf-8-sig"))
    neg_df = normalize_sequence_col(pd.read_csv(neg_csv, encoding="utf-8-sig"))

    # canonical filter (optional cleanup)
    pos_df["sequence"] = pos_df["sequence"].str.replace(r"[^ACDEFGHIKLMNPQRSTVWY]", "", regex=True)
    neg_df["sequence"] = neg_df["sequence"].str.replace(r"[^ACDEFGHIKLMNPQRSTVWY]", "", regex=True)

    # POS: ≤25
    pos_df = filter_by_length(pos_df, min_len, max_len)

    # NEG: crop to match positive length distribution (balanced: same count)
    pos_lengths = pos_df["sequence"].str.len().tolist()
    if not pos_lengths:
        return np.zeros((0, max_len), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    neg_df["seq_len"] = neg_df["sequence"].str.len()
    pool = neg_df[neg_df["seq_len"] >= 1].reset_index(drop=True)
    rng = random.Random(44)

    crops = []
    for L in pos_lengths:
        cand_idx = pool.index[pool["seq_len"] >= L]
        if len(cand_idx) == 0:
            j = rng.choice(pool.index.tolist())
            seq = pool.at[j, "sequence"]
            L_eff = min(len(seq), L)
            start = rng.randrange(0, max(1, len(seq) - L_eff + 1))
            crops.append(seq[start:start + L_eff])
        else:
            j = rng.choice(cand_idx.tolist())
            seq = pool.at[j, "sequence"]
            start = rng.randrange(0, max(1, len(seq) - L + 1))
            crops.append(seq[start:start + L])

    pos_ids = [encode_seq(s) for s in pos_df["sequence"].tolist()]
    neg_ids = [encode_seq(s) for s in crops]
    pos_pad = pad_sequences(pos_ids, max_len)
    neg_pad = pad_sequences(neg_ids, max_len)

    X = np.vstack([pos_pad, neg_pad])
    y = np.hstack([np.ones(len(pos_pad), dtype=np.int64), np.zeros(len(neg_pad), dtype=np.int64)])

    print(f"Unlabelled positives loaded: {len(pos_pad)}")
    print(f"Unlabelled negatives loaded (after length match): {len(neg_pad)}")
    return X, y

amp_x, amp_y = load_unlabelled_xy(DATA_POS_UNLAB, DATA_NEG_UNLAB, MIN_LENGTH, MAX_LENGTH)

# ========= Split into train/test =========
from sklearn.model_selection import train_test_split
x_tr_mic, x_te_mic, y_tr_mic, y_te_mic = train_test_split(
    mic_x, mic_y, test_size=TEST_SIZE, random_state=SEED, stratify=mic_y
)

if len(amp_x):
    amp_x_train, amp_x_test, amp_y_train, amp_y_test = train_test_split(
        amp_x, amp_y, test_size=TEST_SIZE, random_state=SEED, stratify=amp_y
    )
    negatives_x_train = amp_x_train[amp_y_train == 0]
    negatives_x_test  = amp_x_test[amp_y_test  == 0]
else:
    negatives_x_train = np.zeros((0, MAX_LENGTH), dtype=np.int64)
    negatives_x_test  = np.zeros((0, MAX_LENGTH), dtype=np.int64)

x_train = np.concatenate([x_tr_mic, negatives_x_train]) if len(negatives_x_train) else x_tr_mic
y_train = np.concatenate([y_tr_mic.astype(int), np.zeros(len(negatives_x_train), dtype=int)]) if len(negatives_x_train) else y_tr_mic.astype(int)

x_test  = np.concatenate([x_te_mic, negatives_x_test]) if len(negatives_x_test) else x_te_mic
y_test  = np.concatenate([y_te_mic.astype(int), np.zeros(len(negatives_x_test), dtype=int)]) if len(negatives_x_test) else y_te_mic.astype(int)
# ========= MIC dataset balance (before adding AMP positives/negatives) =========
print("==== MIC Dataset Balance ====")
print(f"Train+Validation (MIC only): Positives = {(y_tr_mic==1).sum()}, Negatives = {(y_tr_mic==0).sum()}, Total = {len(y_tr_mic)}")
print(f"Test set (MIC only):        Positives = {(y_te_mic==1).sum()}, Negatives = {(y_te_mic==0).sum()}, Total = {len(y_te_mic)}")
print("================================")

# ========= Unlabelled AMP dataset balance =========
if len(amp_x):
    print("==== Unlabelled AMP Dataset Balance ====")
    print(f"Train+Validation (AMP): Positives = {(amp_y_train==1).sum()}, Negatives = {(amp_y_train==0).sum()}, Total = {len(amp_y_train)}")
    print(f"Test set (AMP):         Positives = {(amp_y_test==1).sum()}, Negatives = {(amp_y_test==0).sum()}, Total = {len(amp_y_test)}")
    print("=======================================")
# ========= Dataset / Model =========
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

class VeltriNet(torch.nn.Module):
    # Emb(21,128) -> Conv1d(64,k=16) -> MaxPool1d(5) -> LSTM(100) -> Dense(1,sigmoid)
    def __init__(self, max_length=25):
        super().__init__()
        self.emb  = torch.nn.Embedding(21, 128, padding_idx=0)
        self.conv = torch.nn.Conv1d(128, 64, kernel_size=16, padding=16//2)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(5)
        self.lstm = torch.nn.LSTM(64, 100, batch_first=True, dropout=0.1)
        self.fc   = torch.nn.Linear(100, 1)
        self.sig  = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.emb(x)               # (N,L,128)
        x = x.transpose(1, 2)         # (N,128,L)
        x = self.relu(self.conv(x))   # (N,64,L)
        x = self.pool(x)              # (N,64,L')
        x = x.transpose(1, 2)         # (N,L',64)
        _, (hn, _) = self.lstm(x)     # hn: (1,N,100)
        x = hn[-1]                    # (N,100)
        return self.sig(self.fc(x)).squeeze(-1)

# ========= Training utilities (match ClassifierLogger: F1-gated early stop & save-best) =========
from sklearn import metrics

def train_with_logger(model, train_loader, val_loader, save_path, epochs=EPOCHS, lr=LR, patience=PATIENCE):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCELoss()

    best_f1 = -1.0
    no_improve = 0
    best_state = None

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            preds = model(xb)
            loss = bce(preds, yb)
            loss.backward()
            opt.step()

        # ----- validate -----
        model.eval()
        with torch.no_grad():
            pv, tv = [], []
            for xb, yb in val_loader:
                pv.append(model(xb.to(DEVICE)).cpu().numpy())
                tv.append(yb.numpy())
            pv = np.concatenate(pv) if pv else np.zeros((0,))
            tv = np.concatenate(tv) if tv else np.zeros((0,))
            vb = (pv > THRESH_BIN).astype(int) if len(pv) else np.zeros((0,), dtype=int)
            f1_val = metrics.f1_score(tv, vb) if len(tv) and len(np.unique(tv)) > 1 else 0.0
            try:
                auc_val = metrics.roc_auc_score(tv, pv)
            except Exception:
                auc_val = float('nan')

        if f1_val > best_f1:
            no_improve = 0
            best_f1 = f1_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), save_path)  # save on each improvement
            print(f"Epoch {ep} - best F1: {round(f1_val,4)}, AUC {round(auc_val,4)}")
        else:
            no_improve += 1
            print(f"Epoch {ep} - current F1: {round(f1_val,4)}, AUC: {round(auc_val,4)}")
            if no_improve >= patience:
                # early stop like the notebook
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict(model, X, bs=10000):
    model.eval()
    dl = torch.utils.data.DataLoader(SeqDataset(X, np.zeros(len(X), dtype=np.float32)), batch_size=bs, shuffle=False)
    out = []
    with torch.no_grad():
        for xb, _ in dl:
            out.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,))

# ========= 10-fold CV with oversampling positives ×5 =========
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

oof_preds = np.zeros((len(x_train),), dtype=np.float32)
test_preds_list = []
mic_test_preds_list = []

fold = 0
for tr_idx, val_idx in cv.split(x_train, y_train):
    fold += 1
    print(f"Fold {fold}")

    X_tr, X_val = x_train[tr_idx], x_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    # oversample positives ×5
    pos_mask = (y_tr == 1)
    X_pos = X_tr[pos_mask]
    if len(X_pos) > 0:
        X_tr_final = np.concatenate([X_tr, np.repeat(X_pos, 5, axis=0)], axis=0)
        y_tr_final = np.concatenate([y_tr, np.ones(len(X_pos)*5, dtype=int)], axis=0)
    else:
        X_tr_final, y_tr_final = X_tr, y_tr

    train_loader = torch.utils.data.DataLoader(SeqDataset(X_tr_final, y_tr_final), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(SeqDataset(X_val, y_val), batch_size=10000, shuffle=False)

    model = VeltriNet(MAX_LENGTH)
    fold_save = os.path.join(SAVE_DIR, f"mic_classifier_fold{fold}_best.pt")
    model = train_with_logger(model, train_loader, val_loader, fold_save, epochs=EPOCHS, lr=LR, patience=PATIENCE)

    # OOF / test preds using restored best
    oof_preds[val_idx] = predict(model, X_val)
    test_preds_list.append(predict(model, x_test))
    mic_test_preds_list.append(predict(model, x_te_mic))

# ========= Aggregate predictions & prints =========
test_avg     = np.mean(np.stack(test_preds_list, axis=0), axis=0).flatten()
mic_test_avg = np.mean(np.stack(mic_test_preds_list, axis=0), axis=0).flatten()

test_avg_bin = np.where(test_avg > THRESH_BIN, 1, 0)
mic_test_bin = np.where(mic_test_avg > THRESH_BIN, 1, 0)
val_preds_bin = np.where(oof_preds > THRESH_BIN, 1, 0)

from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score, confusion_matrix, accuracy_score, matthews_corrcoef

print("Full test classification report")
print(classification_report(y_test, test_avg_bin))

print("Known mic only classification report")
print(classification_report(y_te_mic, mic_test_bin))

print(f"Validation AUC: {roc_auc_score(y_train, oof_preds)}")
print(f"Test AUC: {roc_auc_score(y_test, test_avg)}")

print(f"Validation F1: {f1_score(y_train, val_preds_bin)}")
print(f"Test F1: {f1_score(y_test, test_avg_bin)}")

print(f"Validation sensitivity: {recall_score(y_train, val_preds_bin)}")
print(f"Test sensitivity: {recall_score(y_test, test_avg_bin)}")

tn, fp, fn, tp = confusion_matrix(y_train, val_preds_bin).ravel()
specificity_val  = tn / (tn + fp)
tn, fp, fn, tp = confusion_matrix(y_test, test_avg_bin).ravel()
specificity_test = tn / (tn + fp)
print(f"Validation specificity: {specificity_val}")
print(f"Test specificity: {specificity_test}")

print(f"Validation accuracy: {accuracy_score(y_train, val_preds_bin)}")
print(f"Test accuracy: {accuracy_score(y_test, test_avg_bin)}")

print(f"Validation MCC: {matthews_corrcoef(y_train, val_preds_bin)}")
print(f"Test MCC: {matthews_corrcoef(y_test, test_avg_bin)}")

print(f"Validation Confusion Matrix:\n{confusion_matrix(y_train, val_preds_bin)}")
print(f"Test Confusion Matrix:\n{confusion_matrix(y_test, test_avg_bin)}")

# ========= Final training on full training split, validate on MIC test, save best =========
pos_mask = (y_train == 1)
X_pos = x_train[pos_mask]
X_train_final = np.concatenate([x_train, np.repeat(X_pos, 5, axis=0)], axis=0)
y_train_final = np.concatenate([y_train, np.ones(len(X_pos)*5, dtype=int)], axis=0)

train_loader_full = torch.utils.data.DataLoader(SeqDataset(X_train_final, y_train_final), batch_size=BATCH_SIZE, shuffle=True)
val_loader_mic    = torch.utils.data.DataLoader(SeqDataset(x_te_mic, y_te_mic), batch_size=10000, shuffle=False)

model_final = VeltriNet(MAX_LENGTH)
final_save = os.path.join(SAVE_DIR, "mic_classifier_best.pt")
model_final = train_with_logger(model_final, train_loader_full, val_loader_mic, final_save, epochs=EPOCHS, lr=LR, patience=PATIENCE)

print(f"Saved final best model to: {final_save}")
