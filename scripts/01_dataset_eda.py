"""
scripts/01_dataset_eda.py
=========================
Phase 1 - Step 2: Automated Dataset Acquisition, Quality Control, 
Exploratory Data Analysis (EDA), and 5-Fold Stratified CV Partitioning.

Dataset: AMP Scanner v2 Benchmark Dataset (3,556 sequences)
- 1,778 Positive AMPs (AMP.tr.fa + AMP.te.fa)
- 1,778 Negative non-AMPs / Decoys (DECOY.tr.fa + DECOY.te.fa)

IMPORTANT: Always import sklearn before torch on Windows to prevent DLL/KMP conflicts.
"""

import os
import sys
import urllib.request
import numpy as np
import pandas as pd
from collections import Counter

# Add root folder to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Always import sklearn before torch on Windows
from sklearn.model_selection import StratifiedKFold

# Set random seed
np.random.seed(config.SEED)

print("=" * 70)
print("PHASE 1 - STEP 2: DATASET ACQUISITION & EDA")
print("=" * 70)

# Ensure output directories exist
os.makedirs(config.RAW_DIR, exist_ok=True)
os.makedirs(config.PROCESSED_DIR, exist_ok=True)
os.makedirs(config.FOLDS_DIR, exist_ok=True)
os.makedirs(config.FIGURES_DIR, exist_ok=True)

# Official AMP Scanner v2 GitHub raw dataset URLs
BASE_URL = "https://raw.githubusercontent.com/dan-veltri/amp-scanner-v2/master/original-dataset"
URLS = {
    "pos_train": f"{BASE_URL}/AMP.tr.fa",
    "pos_test":  f"{BASE_URL}/AMP.te.fa",
    "neg_train": f"{BASE_URL}/DECOY.tr.fa",
    "neg_test":  f"{BASE_URL}/DECOY.te.fa",
}

PATHS = {
    "pos_train": os.path.join(config.RAW_DIR, "AMP.tr.fa"),
    "pos_test":  os.path.join(config.RAW_DIR, "AMP.te.fa"),
    "neg_train": os.path.join(config.RAW_DIR, "DECOY.tr.fa"),
    "neg_test":  os.path.join(config.RAW_DIR, "DECOY.te.fa"),
}

# -------------------------------------------------------------------------
# 1. Download Dataset FASTA Files
# -------------------------------------------------------------------------
def download_file(url, target_path):
    if not os.path.exists(target_path):
        print(f"Downloading: {url} -> {target_path}")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
            out_file.write(response.read())
        print("  Download complete.")
    else:
        print(f"  File already exists: {target_path}")

print("\n[1/4] Downloading Official AMP Scanner v2 FASTA Files...")
for key in URLS:
    download_file(URLS[key], PATHS[key])

# -------------------------------------------------------------------------
# 2. Parse & Clean FASTA Files
# -------------------------------------------------------------------------
def parse_fasta(fasta_path, label):
    records = []
    with open(fasta_path, 'r') as f:
        header = None
        seq_parts = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    full_seq = "".join(seq_parts).upper()
                    records.append({'header': header, 'sequence': full_seq, 'label': label})
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            full_seq = "".join(seq_parts).upper()
            records.append({'header': header, 'sequence': full_seq, 'label': label})
    return pd.DataFrame(records)

print("\n[2/4] Parsing and Cleaning Sequences...")
df_pos_tr = parse_fasta(PATHS["pos_train"], label=1)
df_pos_te = parse_fasta(PATHS["pos_test"], label=1)
df_neg_tr = parse_fasta(PATHS["neg_train"], label=0)
df_neg_te = parse_fasta(PATHS["neg_test"], label=0)

df_pos = pd.concat([df_pos_tr, df_pos_te], ignore_index=True)
df_neg = pd.concat([df_neg_tr, df_neg_te], ignore_index=True)

df_all = pd.concat([df_pos, df_neg], ignore_index=True)
print(f"Raw Loaded Records: {len(df_pos)} Positive AMPs | {len(df_neg)} Negative non-AMPs | Total: {len(df_all)}")

# Quality Control: Remove sequences containing non-standard amino acids (B, J, O, U, X, Z)
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid_sequence(seq):
    return all(aa in STANDARD_AA for aa in seq) and len(seq) >= 5

df_clean = df_all[df_all['sequence'].apply(is_valid_sequence)].copy().reset_index(drop=True)
df_clean['seq_len'] = df_clean['sequence'].apply(len)

removed_count = len(df_all) - len(df_clean)
print(f"Quality Control Passed: Retained {len(df_clean)} sequences (Removed {removed_count} invalid/non-standard records).")

# Save combined master CSV
MASTER_CSV = os.path.join(config.PROCESSED_DIR, "amp_benchmark_3556.csv")
df_clean.to_csv(MASTER_CSV, index=False)
print(f"Master Dataset Saved: {MASTER_CSV}")

# -------------------------------------------------------------------------
# 3. Exploratory Data Analysis (EDA)
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS (EDA) REPORT")
print("=" * 70)

# Class Distribution
pos_count = (df_clean['label'] == 1).sum()
neg_count = (df_clean['label'] == 0).sum()
print(f"\n1. Class Distribution:")
print(f"   - Positive (AMPs):     {pos_count} ({pos_count / len(df_clean)*100:.2f}%)")
print(f"   - Negative (non-AMPs): {neg_count} ({neg_count / len(df_clean)*100:.2f}%)")

# Sequence Length Statistics
print(f"\n2. Sequence Length Statistics:")
pos_lens = df_clean[df_clean['label'] == 1]['seq_len']
neg_lens = df_clean[df_clean['label'] == 0]['seq_len']

print(f"   - Overall Length Range: {df_clean['seq_len'].min()} to {df_clean['seq_len'].max()} amino acids")
print(f"   - Positive AMPs Length : Mean = {pos_lens.mean():.2f} ± {pos_lens.std():.2f} (Median = {pos_lens.median():.0f}, Min = {pos_lens.min()}, Max = {pos_lens.max()})")
print(f"   - Negative non-AMPs Length: Mean = {neg_lens.mean():.2f} ± {neg_lens.std():.2f} (Median = {neg_lens.median():.0f}, Min = {neg_lens.min()}, Max = {neg_lens.max()})")

# Amino Acid Frequencies
print(f"\n3. Amino Acid Composition Breakdown:")
def get_aa_freqs(sequences):
    total_aa = sum(len(s) for s in sequences)
    counts = Counter("".join(sequences))
    return {aa: (counts[aa] / total_aa) * 100 for aa in sorted(STANDARD_AA)}

pos_freqs = get_aa_freqs(df_clean[df_clean['label'] == 1]['sequence'])
neg_freqs = get_aa_freqs(df_clean[df_clean['label'] == 0]['sequence'])

aa_summary = pd.DataFrame({
    'AMP Frequency (%)': pos_freqs,
    'non-AMP Frequency (%)': neg_freqs,
    'Difference (AMP - non-AMP)': {aa: pos_freqs[aa] - neg_freqs[aa] for aa in STANDARD_AA}
}).sort_values(by='Difference (AMP - non-AMP)', ascending=False)

print(aa_summary.to_string())

# -------------------------------------------------------------------------
# 4. Generate & Lock 5-Fold Stratified Cross-Validation Folds
# -------------------------------------------------------------------------
print("\n[4/4] Partitioning & Locking 5-Fold Stratified Cross-Validation Splits...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
df_clean['fold'] = -1

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_clean, df_clean['label'])):
    df_clean.loc[val_idx, 'fold'] = fold_idx

# Verify Fold Balance
print("\n5-Fold Split Verification:")
for f in range(5):
    fold_pos = ((df_clean['fold'] == f) & (df_clean['label'] == 1)).sum()
    fold_neg = ((df_clean['fold'] == f) & (df_clean['label'] == 0)).sum()
    print(f"   - Fold {f}: {fold_pos} AMPs + {fold_neg} non-AMPs = {fold_pos + fold_neg} total samples")

# Save Locked Master CSV with Fold Annotations
df_clean.to_csv(MASTER_CSV, index=False)

# Save Individual Fold CSV Files
for f in range(5):
    train_fold = df_clean[df_clean['fold'] != f].reset_index(drop=True)
    val_fold = df_clean[df_clean['fold'] == f].reset_index(drop=True)
    
    train_fold.to_csv(os.path.join(config.FOLDS_DIR, f"train_fold_{f}.csv"), index=False)
    val_fold.to_csv(os.path.join(config.FOLDS_DIR, f"val_fold_{f}.csv"), index=False)

print(f"\nSuccessfully created and locked 5 fold files in '{config.FOLDS_DIR}/'.")
print("=" * 70)
print("PHASE 1 - STEP 2 COMPLETE: READY FOR FEATURE EXTRACTION (STEP 3)!")
print("=" * 70)