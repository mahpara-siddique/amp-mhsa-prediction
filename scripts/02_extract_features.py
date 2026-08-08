"""
scripts/02_extract_features.py
==============================
Phase 1 - Step 3: Feature Extraction Pipeline.

1. Classical Features (AAC + DPC + PhysChem = 430D)
2. ESM-2 (650M) Mean-Pooled Embeddings (1280D)
3. ESM-2 (650M) Per-Residue Tensors ([seq_len x 1280]) saved per sequence ID
"""

import os
import sys

# Ensure root project directory is in Python path FIRST
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from src.features import extract_all_classical

# Always import sklearn before torch on Windows
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("PHASE 1 - STEP 3: FEATURE EXTRACTION PIPELINE")
print("=" * 70)

# Ensure directories exist
os.makedirs(config.FEATURES_DIR, exist_ok=True)
RESIDUE_DIR = os.path.join(config.FEATURES_DIR, "esm2_residue")
os.makedirs(RESIDUE_DIR, exist_ok=True)

# Load Master Dataset
MASTER_CSV = os.path.join(config.PROCESSED_DIR, "amp_benchmark_3556.csv")
df = pd.read_csv(MASTER_CSV)
sequences = df['sequence'].tolist()
labels = df['label'].values
n_samples = len(df)

print(f"\nLoaded {n_samples} benchmark sequences.")

# -------------------------------------------------------------------------
# 1. Classical Feature Extraction (430D)
# -------------------------------------------------------------------------
print("\n[1/3] Extracting Classical Features (AAC + DPC + Physicochemical = 430D)...")
X_classical = extract_all_classical(sequences)
np.save(os.path.join(config.FEATURES_DIR, "classical_430d.npy"), X_classical)
print(f"  Saved Classical Features: {X_classical.shape} -> 'data/features/classical_430d.npy'")

# -------------------------------------------------------------------------
# 2. ESM-2 (650M) Embedding Extraction (Mean + Per-Residue)
# -------------------------------------------------------------------------
print("\n[2/3] Extracting ESM-2 (650M Model) Embeddings on GPU...")
print(f"  Model: {config.ESM_MODEL_NAME}")
print(f"  Device: {config.DEVICE}")

import esm
model, alphabet = esm.pretrained.load_model_and_alphabet(config.ESM_MODEL_NAME)
batch_converter = alphabet.get_batch_converter()
model = model.to(config.DEVICE).eval()

esm2_mean_list = []
batch_size = 8  # Optimal batch size for RTX A4500 (20GB VRAM)

for i in tqdm(range(0, n_samples, batch_size), desc="  Extracting ESM-2"):
    batch_seqs = sequences[i:i+batch_size]
    batch_labels = [(f"seq_{idx}", seq) for idx, seq in enumerate(batch_seqs, start=i)]
    
    _, _, batch_tokens = batch_converter(batch_labels)
    batch_tokens = batch_tokens.to(config.DEVICE)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]  # Shape: [B, L, 1280]
        
        for j, (seq_id, seq) in enumerate(batch_labels):
            seq_idx = i + j
            seq_len = len(seq)
            # Slice off <cls> and <eos> tokens
            residue_emb = token_representations[j, 1:seq_len+1, :].cpu()  # Shape: [seq_len, 1280]
            
            # 1. Save Per-Residue Tensor for novel attention model
            res_path = os.path.join(RESIDUE_DIR, f"seq_{seq_idx}.pt")
            torch.save(residue_emb, res_path)
            
            # 2. Mean Pool over sequence length for baselines
            mean_emb = residue_emb.mean(dim=0).numpy()  # Shape: [1280]
            esm2_mean_list.append(mean_emb)

X_esm2_mean = np.array(esm2_mean_list, dtype=np.float32)
np.save(os.path.join(config.FEATURES_DIR, "esm2_mean_1280d.npy"), X_esm2_mean)
print(f"\n  Saved ESM-2 Mean Embeddings: {X_esm2_mean.shape} -> 'data/features/esm2_mean_1280d.npy'")
print(f"  Saved {n_samples} Per-Residue Tensors -> 'data/features/esm2_residue/'")

# Save Target Labels
np.save(os.path.join(config.FEATURES_DIR, "labels.npy"), labels)
print(f"  Saved Target Labels: {labels.shape} -> 'data/features/labels.npy'")

print("\n" + "=" * 70)
print("PHASE 1 - STEP 3 COMPLETE: READY FOR BASELINE TRAINING (STEP 4-6)!")
print("=" * 70)