# -*- coding: utf-8 -*-
"""Build the generated-peptide supplementary table requested by reviewers.

Run after:
  1. python AmpGEncode.py
  2. python amp_to_22func_pipeline.py

The script merges Stage-1 generation scores, Stage-2 AMP ensemble scores,
Stage-3 ML-A2P probabilities, toxicity-related labels when present, and a
nearest-neighbor identity estimate against the training data.
"""

import argparse
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd


GENERATION_FILES = [
    "Result_1ststage/generated_denovo_ecd_ar.csv",
    "Result_1ststage/generated_analogs_ecd_ar.csv",
]
STAGE2_FILES = [
    "Result_after_2ndstage/generated_denovo_ecd_ar_tagged.csv",
    "Result_after_2ndstage/generated_analogs_ecd_ar_tagged.csv",
]
STAGE3_FILES = [
    "Result_1ststage/generated_denovo_ecd_ar_tagged_amp_22func_pred.csv",
    "Result_1ststage/generated_analogs_ecd_ar_tagged_amp_22func_pred.csv",
    "Final_Result/generated_denovo_ecd_ar_tagged_amp_22func_pred.csv",
    "Final_Result/generated_analogs_ecd_ar_tagged_amp_22func_pred.csv",
]
TRAINING_FILES = [
    ("Dataset/1ststage/unlabelled_positive.csv", None),
    ("Dataset/1ststage/unlabelled_negative.csv", None),
    ("Dataset/1ststage/mic_data.csv", None),
    ("Dataset/1ststage/Uniprot_0_25_train.csv", None),
    ("Dataset/2ndstage/Train_data.csv", None),
    ("Dataset/3rdstage/multilabel_dataset.csv", None),
]
OUTPUT_PATH = "Supplementary_Data_Generated_Peptides.csv"
SEQ_COL_CANDIDATES = ["sequence", "Sequence", "seq", "Seq", "AASequence"]
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def pick_seq_col(df):
    for col in SEQ_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"No sequence column found. Expected one of {SEQ_COL_CANDIDATES}")


def clean_sequence(seq):
    seq = str(seq).strip().upper()
    return "".join(ch for ch in seq if ch in VALID_AA)


def sanitize_label(label):
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(label)).strip("_")


def read_csv_if_exists(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def infer_generation_mode(path):
    name = os.path.basename(path).lower()
    if "denovo" in name or "de_novo" in name:
        return "de_novo"
    if "analog" in name:
        return "analog"
    return "generated"


def numeric_or_default(df, col, default=0.0):
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float32")
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype("float32")


def load_generated(files):
    frames = []
    for path in files:
        df = read_csv_if_exists(path)
        if df is None:
            continue
        seq_col = pick_seq_col(df)
        df = df.copy()
        if seq_col != "sequence":
            df = df.rename(columns={seq_col: "sequence"})
        df["sequence"] = df["sequence"].map(clean_sequence)
        df = df[df["sequence"].str.len() > 0].copy()
        if "generation_mode" not in df.columns:
            df["generation_mode"] = infer_generation_mode(path)
        if "name" not in df.columns:
            df["name"] = [f"{df['generation_mode'].iloc[0]}_{i:05d}" for i in range(len(df))]
        if "reward_score" not in df.columns:
            amp = numeric_or_default(df, "amp_prob")
            mic = numeric_or_default(df, "mic_prob_low")
            df["reward_score"] = np.clip(amp + mic, -1.0, 1.0)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            "No Stage-1 generated peptide CSVs were found. Run AmpGEncode.py first."
        )
    return pd.concat(frames, ignore_index=True)


def load_stage2(files):
    frames = []
    for path in files:
        df = read_csv_if_exists(path)
        if df is None:
            continue
        seq_col = pick_seq_col(df)
        df = df.copy()
        if seq_col != "sequence":
            df = df.rename(columns={seq_col: "sequence"})
        df["sequence"] = df["sequence"].map(clean_sequence)
        keep = ["sequence"]
        for col in ["ensemble_softscore", "ensemble_amp"]:
            if col in df.columns:
                keep.append(col)
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame(columns=["sequence", "ensemble_softscore", "ensemble_amp"])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["sequence"], keep="first")


def load_stage3(files):
    frames = []
    for path in files:
        df = read_csv_if_exists(path)
        if df is None:
            continue
        seq_col = pick_seq_col(df)
        df = df.copy()
        if seq_col != "sequence":
            df = df.rename(columns={seq_col: "sequence"})
        df["sequence"] = df["sequence"].map(clean_sequence)
        prob_cols = [col for col in df.columns if col.endswith("_prob")]
        binary_cols = [
            col for col in df.columns
            if col not in set(prob_cols + ["sequence"]) and col in df.columns
        ]
        keep = ["sequence"] + prob_cols + binary_cols
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame(columns=["sequence"])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["sequence"], keep="first")


def load_training_sequences(training_files):
    seqs = []
    for path, seq_col in training_files:
        df = read_csv_if_exists(path)
        if df is None:
            continue
        col = seq_col or pick_seq_col(df)
        seqs.extend(df[col].map(clean_sequence).tolist())
    seqs = sorted({seq for seq in seqs if seq})
    if not seqs:
        raise FileNotFoundError("No training sequences were found for nearest-neighbor search.")
    return seqs


def build_kmer_index(seqs, k=3):
    index = defaultdict(list)
    for idx, seq in enumerate(seqs):
        kmers = {seq[i:i + k] for i in range(max(0, len(seq) - k + 1))}
        for kmer in kmers:
            index[kmer].append(idx)
    length_index = defaultdict(list)
    for idx, seq in enumerate(seqs):
        length_index[len(seq)].append(idx)
    return index, length_index


@lru_cache(maxsize=200000)
def edit_distance(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def identity_pct(query, target):
    denom = max(len(query), len(target), 1)
    return 100.0 * (1.0 - edit_distance(query, target) / denom)


def candidate_indices(query, seqs, kmer_index, length_index, k=3, limit=1000):
    candidates = set()
    counts = Counter()
    for length in range(max(1, len(query) - 2), len(query) + 3):
        candidates.update(length_index.get(length, []))
    if len(query) >= k:
        for i in range(len(query) - k + 1):
            for idx in kmer_index.get(query[i:i + k], []):
                counts[idx] += 1
        candidates.update(idx for idx, _ in counts.most_common(limit))
    if not candidates:
        candidates.update(range(min(limit, len(seqs))))
    if len(candidates) > limit:
        candidates = set(sorted(
            candidates,
            key=lambda idx: (-counts.get(idx, 0), abs(len(seqs[idx]) - len(query)))
        )[:limit])
    return candidates


def nearest_neighbors(queries, training_seqs, candidate_limit=1000):
    kmer_index, length_index = build_kmer_index(training_seqs)
    nearest_seq = []
    nearest_id = []
    for query in queries:
        best_seq = ""
        best_identity = -1.0
        for idx in candidate_indices(query, training_seqs, kmer_index, length_index, limit=candidate_limit):
            ident = identity_pct(query, training_seqs[idx])
            if ident > best_identity:
                best_identity = ident
                best_seq = training_seqs[idx]
                if best_identity == 100.0:
                    break
        nearest_seq.append(best_seq)
        nearest_id.append(round(best_identity, 3))
    return nearest_seq, nearest_id


def add_toxicity_columns(df):
    for label in ["Hemolytic", "Cytotoxic"]:
        prob_col = f"{label}_prob"
        pred_col = f"{label.lower()}_prediction"
        if prob_col in df.columns:
            df[f"{label.lower()}_probability"] = df[prob_col]
            df[pred_col] = (pd.to_numeric(df[prob_col], errors="coerce") >= 0.5).astype("Int64")
        else:
            df[f"{label.lower()}_probability"] = np.nan
            df[pred_col] = pd.Series([pd.NA] * len(df), dtype="Int64")
    return df


def build_supplementary_table(args):
    generated = load_generated(args.generation_files)
    stage2 = load_stage2(args.stage2_files)
    stage3 = load_stage3(args.stage3_files)

    table = generated.merge(stage2, on="sequence", how="left", suffixes=("", "_stage2"))
    table = table.merge(stage3, on="sequence", how="left", suffixes=("", "_stage3"))

    training_seqs = load_training_sequences(TRAINING_FILES)
    nn_seq, nn_id = nearest_neighbors(
        table["sequence"].tolist(),
        training_seqs,
        candidate_limit=args.candidate_limit,
    )
    table["nearest_training_sequence"] = nn_seq
    table["nearest_neighbor_identity_to_training_pct"] = nn_id

    table = add_toxicity_columns(table)

    rename = {
        "name": "peptide_id",
        "amp_prob": "amp_probability",
        "mic_prob_low": "mic_class_probability",
        "ensemble_softscore": "ensemble_screening_score",
    }
    table = table.rename(columns=rename)

    prob_cols = sorted(
        col for col in table.columns
        if col.endswith("_prob") and col not in {"amp_prob", "mic_prob_low"}
    )
    ml_a2p_renames = {
        col: f"ml_a2p_{sanitize_label(col[:-5]).lower()}_probability"
        for col in prob_cols
    }
    table = table.rename(columns=ml_a2p_renames)

    front_cols = [
        "peptide_id",
        "sequence",
        "generation_mode",
        "seed",
        "reward_score",
        "amp_probability",
        "mic_class_probability",
        "ensemble_screening_score",
        "ensemble_amp",
    ]
    ml_cols = sorted(col for col in table.columns if col.startswith("ml_a2p_"))
    toxicity_cols = [
        "hemolytic_probability",
        "hemolytic_prediction",
        "cytotoxic_probability",
        "cytotoxic_prediction",
    ]
    nn_cols = ["nearest_training_sequence", "nearest_neighbor_identity_to_training_pct"]
    property_cols = [
        col for col in ["length", "hydrophobicity", "hydrophobic_moment", "charge_pH7_4", "pI"]
        if col in table.columns
    ]
    ordered = [col for col in front_cols + property_cols + ml_cols + toxicity_cols + nn_cols if col in table.columns]
    ordered += [col for col in table.columns if col not in ordered]
    table = table[ordered]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    table.to_csv(args.output, index=False)
    return table


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--candidate-limit", type=int, default=1000)
    parser.add_argument("--generation-files", nargs="+", default=GENERATION_FILES)
    parser.add_argument("--stage2-files", nargs="+", default=STAGE2_FILES)
    parser.add_argument("--stage3-files", nargs="+", default=STAGE3_FILES)
    return parser.parse_args()


def main():
    args = parse_args()
    table = build_supplementary_table(args)
    print(f"Wrote {args.output} with {len(table)} generated peptides and {len(table.columns)} columns.")


if __name__ == "__main__":
    main()
