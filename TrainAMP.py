# amp_classifier_pytorch_noconv.py
# Spyder-ready PyTorch reimplementation of your NoConv AMP classifier notebook.

import os
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================
# Config / seeds
# =========================
def set_seed(seed=36):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(36)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data / vocab
VOCAB = list('ACDEFGHIKLMNPQRSTVWY')  # 20 AA
PAD_IDX = 0
AA2IDX = {aa: i+1 for i, aa in enumerate(VOCAB)}  # 1..20, 0=PAD
IDX2AA = {v: k for k, v in AA2IDX.items()}

MIN_LENGTH = 0
MAX_LENGTH = 25
VOCAB_SIZE = 21  # 20 + PAD

# Model / training hyperparams (match Keras)
EMBED_DIM = 128
LSTM1_HIDDEN = 64
LSTM2_HIDDEN = 100
POOL_KERNEL = 5
LSTM_DROPOUT = 0.1

EPOCHS_CV = 100           # your notebook used 100 in CV
EPOCHS_FINAL = 42         # your notebook used 42 in final fit
BATCH_SIZE = 128
LR = 1e-3
PATIENCE = 10
N_SPLITS = 10
SHOW_PLOTS = True

SAVE_DIR_FOLDS = "models"   # mirrors your 'amp_classifier_training/'
os.makedirs(SAVE_DIR_FOLDS, exist_ok=True)
FINAL_MODEL_PATH = "models/amp_classifier.pt"
os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)

# =========================
# Utils: encode/pad/lengths
# =========================
def to_indices(seqs):
    return [[AA2IDX[aa] for aa in s] for s in seqs]

def pad_to_length(idxs, max_len=MAX_LENGTH):
    arr = np.zeros((len(idxs), max_len), dtype=np.int64)
    for i, seq in enumerate(idxs):
        L = min(len(seq), max_len)
        if L > 0:
            arr[i, :L] = np.array(seq[:L], dtype=np.int64)
    return arr

def fast_lengths(padded_nd):
    return (padded_nd != 0).sum(axis=1)

# =========================
# Negative subsequence balancing (AMPDataManager behavior)
# =========================
def _length_probs(positive_lengths):
    counts = defaultdict(lambda: 1)  # Laplace-like smoothing
    for L in positive_lengths:
        counts[L] += 1
    total = sum(counts.values())
    keys = np.array(sorted(counts.keys()))
    probs = np.array([counts[k] / total for k in keys])
    return keys, probs

def _draw_subsequences(neg_df, target_lengths, seq_col="Sequence", name_col="Name"):
    rng = np.random.default_rng(44)
    df = neg_df.copy()
    df["Sequence length"] = df[seq_col].str.len().astype(int)
    df = df.sort_values("Sequence length", ascending=False).reset_index(drop=True)
    tgt = sorted(list(target_lengths), reverse=True)

    out_names, out_seqs = [], []
    for (idx, row), new_len in zip(df.iterrows(), tgt):
        seq = row[seq_col]
        L = row["Sequence length"]
        if new_len >= L:
            new_seq = seq
        else:
            start = rng.integers(0, L - new_len + 1)
            new_seq = seq[start:start + new_len]
        out_names.append(row[name_col] if name_col in df.columns else f"neg_{idx}")
        out_seqs.append(new_seq)
    return pd.DataFrame({name_col: out_names, seq_col: out_seqs})

def balance_negatives_to_positives(pos_df, neg_df):
    # filter positives by length
    pos_df = pos_df.loc[(pos_df["Sequence"].str.len() >= MIN_LENGTH) &
                        (pos_df["Sequence"].str.len() <= MAX_LENGTH)].reset_index(drop=True)
    # build positive length distribution
    pos_lengths = pos_df["Sequence"].str.len().astype(int).tolist()
    keys, probs = _length_probs(pos_lengths)
    k = len(pos_lengths)  # balanced count
    target_lengths = np.random.choice(keys, size=k, p=probs, replace=True)
    # crop negatives to these lengths
    neg_cropped = _draw_subsequences(neg_df, target_lengths, seq_col="Sequence", name_col="Name")
    return pos_df, neg_cropped

# =========================
# Dataset / Model
# =========================
class AmpDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.int64)
        self.y = y.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

class NoConvAmpClassifier(nn.Module):
    """
    Embedding -> LSTM(64, seq) -> MaxPool1d(k=5) -> LSTM(100) -> Dense(sigmoid)
    Faithful to your Keras NoConvAMPClassifier.
    """
    def __init__(self,
                 vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 lstm1_hidden=LSTM1_HIDDEN, lstm2_hidden=LSTM2_HIDDEN,
                 pool_kernel=POOL_KERNEL, lstm_dropout=LSTM_DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm1 = nn.LSTM(embed_dim, lstm1_hidden, batch_first=True, dropout=lstm_dropout)
        self.pool = nn.MaxPool1d(pool_kernel)  # time pooling
        self.lstm2 = nn.LSTM(lstm1_hidden, lstm2_hidden, batch_first=True, dropout=lstm_dropout)
        self.fc = nn.Linear(lstm2_hidden, 1)
        # dense-input path if ever needed (one-hot -> embedding projection)
        self.dense_emb = nn.Linear(21, embed_dim, bias=False)

    def forward_indices(self, x_idx):
        x = self.embedding(x_idx)         # (B, L, E)
        x1, _ = self.lstm1(x)             # (B, L, H1)
        x1p = x1.permute(0, 2, 1)         # (B, H1, L)
        x1p = self.pool(x1p)              # (B, H1, L')
        x1p = x1p.permute(0, 2, 1)        # (B, L', H1)
        x2, (h, _) = self.lstm2(x1p)      # h: (1, B, H2)
        h = h[-1]                          # (B, H2)
        return torch.sigmoid(self.fc(h))   # (B, 1)

    def forward_dense(self, x_dense):
        x = self.dense_emb(x_dense)        # (B, L, E)
        x1, _ = self.lstm1(x)
        x1p = x1.permute(0, 2, 1)
        x1p = self.pool(x1p)
        x1p = x1p.permute(0, 2, 1)
        x2, (h, _) = self.lstm2(x1p)
        h = h[-1]
        return torch.sigmoid(self.fc(h))

    def forward(self, x):
        if x.dtype in (torch.long, torch.int64):
            return self.forward_indices(x)
        if x.dim() == 3 and x.size(-1) == 21:
            return self.forward_dense(x)
        raise ValueError("Input must be Long indices (B,L) or Float one-hot (B,L,21).")

# =========================
# Training (early stopping on F1) + save best per fold
# =========================
def train_one_fold(model, train_loader, val_loader, split_index,
                   epochs=EPOCHS_CV, lr=LR, patience=PATIENCE, save_dir=SAVE_DIR_FOLDS):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    best_f1, no_improve = -1.0, 0
    best_state = None
    save_path = os.path.join(save_dir, f"{split_index}.pt")  # mirrors '{split_index}.h5'

    for ep in range(1, epochs + 1):
        # train
        model.train(); losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            prob = model(xb)
            loss = criterion(prob, yb)
            loss.backward(); optimizer.step()
            losses.append(loss.item())
        train_loss = np.mean(losses) if losses else 0.0

        # validate
        model.eval(); preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                p = model(xb).cpu().numpy().ravel()
                preds.extend(p); labels.extend(yb.numpy().ravel())
        preds = np.array(preds); labels = np.array(labels)
        binp = (preds > 0.5).astype(int)
        f1 = metrics.f1_score(labels, binp)
        auc = metrics.roc_auc_score(labels, preds)
        print(f"Train on {len(train_loader.dataset)} samples, validate on {len(val_loader.dataset)} samples")
        print(f"Epoch {ep}/{epochs}  - loss: {train_loss:.4f}  - val AUC: {auc:.4f}")

        # mimic your logger: print best F1 and save weights on improvement
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
            print(f"Epoch {ep} - best F1: {f1:.4f}, AUC {auc:.4f}  [saved -> {save_path}]")
            no_improve = 0
        else:
            no_improve += 1
            print(f"Epoch {ep} - current F1: {f1:.4f}, AUC {auc:.4f}")
            if no_improve >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        torch.save(model.state_dict(), save_path)
    return model, save_path

# =========================
# Main pipeline (mirrors your notebook cells)
# =========================
def main():
    # --- Load raw CSVs ---
    pos_raw = pd.read_csv('Dataset/1ststage/unlabelled_positive.csv')
    neg_raw = pd.read_csv('Dataset/1ststage/unlabelled_negative.csv')

    # --- Balance negatives by cropping subsequences to match positive lengths ---
    pos_df, neg_df = balance_negatives_to_positives(pos_raw, neg_raw)

    # Plot distributions (equalize=True) like your cell [4]
    if SHOW_PLOTS:
        fig, (ax2, ax3) = plt.subplots(figsize=(12, 6), ncols=2)
        ax2.hist([len(s) for s in pos_df["Sequence"]], bins=30, density=True)
        ax3.hist([len(s) for s in neg_df["Sequence"]], bins=30, density=True)
        ax2.set_title("Positive"); ax3.set_title("Negative")
        plt.show()

    # --- Join datasets (labels) and encode/pad ---
    pos_ids = pad_to_length(to_indices(pos_df["Sequence"].tolist()))
    neg_ids = pad_to_length(to_indices(neg_df["Sequence"].tolist()))
    pos_y = np.ones(len(pos_ids), dtype=np.int64)
    neg_y = np.zeros(len(neg_ids), dtype=np.int64)
    x = np.concatenate([pos_ids, neg_ids], axis=0)
    y = np.concatenate([pos_y,  neg_y],  axis=0)

    # --- Train/test split ---
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=36)

    # --- KFold ---
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=36)

    # --- CV loop (save each split best) ---
    oof_preds = np.zeros((len(x_train), 1), dtype=float)
    test_preds = []

    for split_index, (tr_idx, va_idx) in enumerate(cv.split(x_train, y_train)):
        print(f"\nFold {split_index}")

        model = NoConvAmpClassifier().to(DEVICE)
        tr_ds = AmpDataset(x_train[tr_idx], y_train[tr_idx])
        va_ds = AmpDataset(x_train[va_idx], y_train[va_idx])
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

        model, _ = train_one_fold(model, tr_loader, va_loader, split_index)

        # Predict test for this fold (batch_size=64 like your code)
        model.eval()
        test_loader = DataLoader(AmpDataset(x_test, y_test), batch_size=64, shuffle=False)
        fold_test = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(DEVICE)
                p = model(xb).cpu().numpy().ravel()
                fold_test.extend(p)
        test_preds.append(np.array(fold_test))

        # OOF preds for this fold
        val_loader2 = DataLoader(va_ds, batch_size=64, shuffle=False)
        fold_val = []
        with torch.no_grad():
            for xb, _ in val_loader2:
                xb = xb.to(DEVICE)
                p = model(xb).cpu().numpy().ravel()
                fold_val.extend(p)
        oof_preds[va_idx, 0] = np.array(fold_val)

    # --- Aggregate metrics like your notebook cells [8]–[16] ---
    test_avg = np.mean(np.asarray(test_preds), axis=0).flatten()
    test_avg_bin = (test_avg > 0.5).astype(int)
    val_preds_bin = (oof_preds > 0.5).astype(int)

    print(f"\nValidation AUC: {metrics.roc_auc_score(y_train, oof_preds)}")
    print(f"Test AUC: {metrics.roc_auc_score(y_test, test_avg)}")

    print(f"\nValidation F1: {metrics.f1_score(y_train, val_preds_bin)}")
    print(f"Test F1: {metrics.f1_score(y_test, test_avg_bin)}")

    print(f"\nValidation sensitivity: {metrics.recall_score(y_train, val_preds_bin)}")
    print(f"Test sensitivity: {metrics.recall_score(y_test, test_avg_bin)}")

    tn, fp, fn, tp = metrics.confusion_matrix(y_train, val_preds_bin).ravel()
    specificity_val = tn / (tn + fp)
    tn2, fp2, fn2, tp2 = metrics.confusion_matrix(y_test, test_avg_bin).ravel()
    specificity_test = tn2 / (tn2 + fp2)
    print(f"\nValidation specificity: {specificity_val}")
    print(f"Test specificity: {specificity_test}")

    print(f"\nValidation accuracy: {metrics.accuracy_score(y_train, val_preds_bin)}")
    print(f"Test accuracy: {metrics.accuracy_score(y_test, test_avg_bin)}")

    print(f"\nValidation MCC: {metrics.matthews_corrcoef(y_train, val_preds_bin)}")
    print(f"Test MCC: {metrics.matthews_corrcoef(y_test, test_avg_bin)}")

    print(f"\nValidation Confusion Matrix:\n{metrics.confusion_matrix(y_train, val_preds_bin)}")
    print(f"\nTest Confusion Matrix:\n{metrics.confusion_matrix(y_test, test_avg_bin)}")

    # --- Final single fit on full train (like your last cell) and save ---
    final_model = NoConvAmpClassifier().to(DEVICE)
    final_loader = DataLoader(AmpDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    # simple training loop for EPOCHS_FINAL
    opt = torch.optim.Adam(final_model.parameters(), lr=LR)
    crit = nn.BCELoss()
    final_model.train()
    for ep in range(1, EPOCHS_FINAL + 1):
        losses = []
        for xb, yb in final_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
            opt.zero_grad()
            p = final_model(xb)
            loss = crit(p, yb)
            loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"FinalFit Epoch {ep}/{EPOCHS_FINAL} - loss: {np.mean(losses):.4f}")

    torch.save(final_model.state_dict(), FINAL_MODEL_PATH)
    print(f"\nSaved final AMP classifier to {FINAL_MODEL_PATH}")

if __name__ == "__main__":
    main()
