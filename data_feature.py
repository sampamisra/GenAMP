# -*- coding: utf-8 -*-
"""
Embedding helpers:
- AAI_embedding   -> (N, L, 531)
- onehot_embedding-> (N, L, 21)  # 20 AA + 1 unknown (includes '-')
- BLOSUM62_embedding -> (N, L, 23)
- PAAC_embedding  -> (N, L, 3)

All outputs are float32 and robust to uncommon letters (U,O,B,Z,J,...).
"""

import numpy as np
import pandas as pd
import torch
import vocab

# ========================== utilities ==========================

def _pad_or_trim(rows, max_len, dim):
    """
    rows: list of 1D arrays length 'dim'
    returns: (max_len, dim) float32
    """
    if len(rows) == 0:
        return np.zeros((max_len, dim), dtype=np.float32)
    arr = np.asarray(rows, dtype=np.float32)
    L = arr.shape[0]
    if L >= max_len:
        return arr[:max_len].astype(np.float32, copy=False)
    out = np.zeros((max_len, dim), dtype=np.float32)
    out[:L] = arr
    return out

# ========================== AAI (531) ==========================

def AAI_embedding(seq, max_len=200):
    with open('data/AAindex.txt', 'r', encoding='utf-8', errors='ignore') as f:
        text = [ln for ln in f.read().split('\n') if ln.strip() != '']
    cha = [c for c in text[0].split('\t') if c != ''][1:]
    index = []
    for line in text[1:]:
        temp = [t for t in line.split('\t') if t != ''][1:]
        index.append([float(x) for x in temp])
    index = np.asarray(index, dtype=np.float32)            # (531, len(cha))
    AAI_dict = {cha[j]: index[:, j] for j in range(len(cha))}
    AAI_dict['X'] = np.zeros(531, dtype=np.float32)
    for k in ['U','O','B','Z','J','*','-','?']:
        AAI_dict.setdefault(k, AAI_dict['X'])

    out = []
    for s in seq:
        rows = [AAI_dict.get(ch, AAI_dict['X']) for ch in str(s).upper()]
        out.append(_pad_or_trim(rows, max_len, 531))
    return torch.from_numpy(np.asarray(out, dtype=np.float32))

# ========================== PAAC (3) ==========================

def PAAC_embedding(seq, max_len=200):
    with open('data/PAAC.txt', 'r', encoding='utf-8', errors='ignore') as f:
        text = [ln for ln in f.read().split('\n') if ln.strip() != '']
    cha = [c for c in text[0].split('\t') if c != ''][1:]
    index = []
    for line in text[1:]:
        temp = [t for t in line.split('\t') if t != ''][1:]
        index.append([float(x) for x in temp])
    index = np.asarray(index, dtype=np.float32)            # (3, len(cha))
    AAI_dict = {cha[j]: index[:, j] for j in range(len(cha))}
    AAI_dict['X'] = np.zeros(3, dtype=np.float32)
    for k in ['U','O','B','Z','J','*','-','?']:
        AAI_dict.setdefault(k, AAI_dict['X'])

    out = []
    for s in seq:
        rows = [AAI_dict.get(ch, AAI_dict['X']) for ch in str(s).upper()]
        out.append(_pad_or_trim(rows, max_len, 3))
    return torch.from_numpy(np.asarray(out, dtype=np.float32))

# ========================== BLOSUM62 (23) ==========================

def BLOSUM62_embedding(seq, max_len=200):
    with open('data/blosum62.txt', 'r', encoding='utf-8', errors='ignore') as f:
        text = [ln for ln in f.read().split('\n') if ln.strip() != '']
    cha = [c for c in text[0].split(' ') if c != '']       # expects 23 tokens
    index = []
    for line in text[1:]:
        temp = [t for t in line.split(' ') if t != '']
        index.append([float(x) for x in temp])
    index = np.asarray(index, dtype=np.float32)            # (23, 23)
    BLOSUM62_dict = {cha[j]: index[:, j] for j in range(len(cha))}
    BLOSUM62_dict['X'] = np.zeros(23, dtype=np.float32)
    for k in ['U','O','B','Z','J','*','-','?']:
        BLOSUM62_dict.setdefault(k, BLOSUM62_dict['X'])

    out = []
    for s in seq:
        rows = [BLOSUM62_dict.get(ch, BLOSUM62_dict['X']) for ch in str(s).upper()]
        out.append(_pad_or_trim(rows, max_len, 23))
    return torch.from_numpy(np.asarray(out, dtype=np.float32))

# ========================== One-hot (21) ==========================

def onehot_embedding(seq, max_len=200):
    """
    21-dim one-hot: 20 canonical amino acids + 1 'unknown' (includes '-' and any non-canonical).
    Matches model expectation d_input[1] = 21.
    """
    alphabet = 'ARNDCQEGHILKMFPSTWYV'  # 20 canonical
    idx = {ch: i for i, ch in enumerate(alphabet)}
    unknown_idx = 20                     # position of 'X'-like bucket
    C = 21

    out = []
    for s in seq:
        s = str(s).upper()
        rows = []
        for ch in s:
            v = np.zeros((C,), dtype=np.float32)
            v[idx.get(ch, unknown_idx)] = 1.0   # '-' and any non-canonical go to unknown_idx
            rows.append(v)
        out.append(_pad_or_trim(rows, max_len, C))
    return torch.from_numpy(np.asarray(out, dtype=np.float32))

# ========================== Encoding helper (unchanged) ==========================

def seq2encoding(seq, max_len=200):
    seq_list = []
    for temp in seq:
        temp = list(temp)
        if len(temp) < max_len:
            temp = temp + ['X'] * (max_len - len(temp))
        else:
            temp = temp[:max_len]
        seq_list.append(temp)
    df = pd.DataFrame(seq_list)
    encoding = df.replace(vocab.AMINO_ACID_INDEX)
    return encoding.values.astype(int)

class Met:
    pass
