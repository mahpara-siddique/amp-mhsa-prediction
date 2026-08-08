"""
scripts/04_train_proposed.py
=============================
Phase 3: AMP-MHSA — Lightweight Bio-Informed Architecture
1. Full Model 5-Fold CV (25 epochs)
2. 12-Experiment Ablation Study (5-Fold CV × 15 epochs)
3. Publication Figures
"""

import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, recall_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score, f1_score, roc_curve)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import config
from src.models.proposed_model import AMP_MHSA, FocalLoss, compute_all_residue_features

np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)

print("=" * 70)
print("PHASE 3: AMP-MHSA — LIGHTWEIGHT BIO-INFORMED ARCHITECTURE")
print("  Components: Bio Attention Bias | Helical PE | μH Feature | Wimley-White")
print("  Extra bio params: ~3,009  (vs ~2.5M in heavy cross-attention)")
print("=" * 70)

MASTER_CSV = os.path.join(config.PROCESSED_DIR, "amp_benchmark_3556.csv")
df_master = pd.read_csv(MASTER_CSV)
sequences = df_master['sequence'].tolist()
labels = df_master['label'].values
folds_array = df_master['fold'].values
RESIDUE_DIR = os.path.join(config.FEATURES_DIR, "esm2_residue")

# =========================================================================
# Dataset
# =========================================================================

class AMPResidueDataset(Dataset):
    def __init__(self, indices, labels_array, sequences_list):
        self.indices = indices
        self.labels = labels_array[indices]
        self.sequences = [sequences_list[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        seq_idx = self.indices[idx]
        esm_tensor = torch.load(os.path.join(RESIDUE_DIR, f"seq_{seq_idx}.pt"))
        features = compute_all_residue_features(self.sequences[idx])  # [L, 12]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return esm_tensor, features, label

def collate_fn(batch):
    esm_list, feat_list, labels = zip(*batch)
    lengths = [len(t) for t in esm_list]
    padded_esm = pad_sequence(esm_list, batch_first=True, padding_value=0.0)
    padded_feat = pad_sequence(feat_list, batch_first=True, padding_value=0.0)
    mask = torch.zeros(len(esm_list), max(lengths), dtype=torch.float32)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0
    return padded_esm, padded_feat, mask, torch.stack(labels)

# =========================================================================
# Metrics
# =========================================================================

def calc_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
    return acc, sens, spec, mcc, auc, f1

# =========================================================================
# Train & Evaluate
# =========================================================================

def train_eval_fold(fold_num, pooling_mode="fused", loss_type="bce", dropout=0.4,
                    use_layernorm=True, use_bio_bias=True, use_helix_pe=True,
                    use_wimley=True, use_moment=True, epochs=25):

    val_idx = np.where(folds_array == fold_num)[0]
    train_idx = np.where(folds_array != fold_num)[0]

    train_loader = DataLoader(AMPResidueDataset(train_idx, labels, sequences),
                              batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(AMPResidueDataset(val_idx, labels, sequences),
                            batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = AMP_MHSA(
        embed_dim=1280, phys_dim=12, num_heads=4,
        pooling_mode=pooling_mode, dropout=dropout, use_layernorm=use_layernorm,
        use_bio_bias=use_bio_bias, use_helix_pe=use_helix_pe,
        use_wimley=use_wimley, use_moment=use_moment
    ).to(config.DEVICE)

    criterion = FocalLoss(gamma=2.0) if loss_type == "focal" else nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        for esm_b, feat_b, mask_b, y_b in train_loader:
            esm_b, feat_b, mask_b, y_b = (esm_b.to(config.DEVICE), feat_b.to(config.DEVICE),
                                           mask_b.to(config.DEVICE), y_b.to(config.DEVICE))
            optimizer.zero_grad()
            logits, _ = model(esm_b, feat_b, mask_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for esm_b, feat_b, mask_b, y_b in val_loader:
            esm_b, feat_b, mask_b = (esm_b.to(config.DEVICE), feat_b.to(config.DEVICE),
                                      mask_b.to(config.DEVICE))
            logits, _ = model(esm_b, feat_b, mask_b)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_targets.extend(y_b.numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    return all_targets, (all_probs >= 0.5).astype(int), all_probs

# =========================================================================
# 1. Full Model (5-Fold CV, 25 epochs)
# =========================================================================
print("\n[1/3] Full AMP-MHSA (Bio Bias + Helix PE + μH + Wimley) — 5-Fold CV, 25 epochs...")

fold_metrics = []
all_y_true, all_y_prob = [], []

for f in range(5):
    print(f"  Fold {f+1}/5...", end=" ")
    y_true, y_pred, y_prob = train_eval_fold(f)
    m = calc_metrics(y_true, y_pred, y_prob)
    fold_metrics.append(m)
    all_y_true.extend(y_true); all_y_prob.extend(y_prob)
    print(f"Acc={m[0]*100:.2f}% | MCC={m[3]:.4f} | AUROC={m[4]:.4f}")

fold_metrics = np.array(fold_metrics)
m_means = fold_metrics.mean(axis=0)
m_stds = fold_metrics.std(axis=0)

print(f"\n  ★ FULL MODEL 5-Fold Results:")
for name, i in [("Accuracy", 0), ("Sensitivity", 1), ("Specificity", 2),
                ("MCC", 3), ("AUROC", 4), ("F1-Score", 5)]:
    if i <= 2 or i == 5:
        print(f"   {name:12s}: {m_means[i]*100:.2f}% ± {m_stds[i]*100:.2f}%")
    else:
        print(f"   {name:12s}: {m_means[i]:.4f} ± {m_stds[i]:.4f}")

# =========================================================================
# 2. 12-Experiment Ablation (5-Fold CV × 15 epochs)
# =========================================================================
print("\n[2/3] 12-Experiment Ablation (5-Fold CV × 15 epochs each)...\n")

#                    Name                   pool       loss   dp   ln   bio  helix wim  mom
ablations = [
    ("Exp01_Full_Model",       "fused",     "bce",  0.4, True, True, True, True, True),
    ("Exp02_No_Bio_Bias",      "fused",     "bce",  0.4, True, False,True, True, True),
    ("Exp03_No_Helix_PE",      "fused",     "bce",  0.4, True, True, False,True, True),
    ("Exp04_No_Wimley",        "fused",     "bce",  0.4, True, True, True, False,True),
    ("Exp05_No_Moment",        "fused",     "bce",  0.4, True, True, True, True, False),
    ("Exp06_Vanilla_Baseline", "fused",     "bce",  0.4, True, False,False,False,False),
    ("Exp07_Attn_Only",        "attn_only", "bce",  0.4, True, True, True, True, True),
    ("Exp08_Mean_Only",        "mean_only", "bce",  0.4, True, False,False,False,False),
    ("Exp09_Focal_Loss",       "fused",     "focal",0.4, True, True, True, True, True),
    ("Exp10_No_LayerNorm",     "fused",     "bce",  0.4, False,True, True, True, True),
    ("Exp11_Dropout_0.2",      "fused",     "bce",  0.2, True, True, True, True, True),
    ("Exp12_Dropout_0.5",      "fused",     "bce",  0.5, True, True, True, True, True),
]

ablation_results = []

for name, pool, loss_t, dp, ln, bio, helix, wim, mom in ablations:
    f_ms = []
    for f in range(5):
        yt, yp, yb = train_eval_fold(
            f, pooling_mode=pool, loss_type=loss_t, dropout=dp,
            use_layernorm=ln, use_bio_bias=bio, use_helix_pe=helix,
            use_wimley=wim, use_moment=mom, epochs=15
        )
        f_ms.append(calc_metrics(yt, yp, yb))
    f_ms = np.array(f_ms)
    means = f_ms.mean(axis=0)

    ablation_results.append({
        "Experiment": name,
        "Pooling": pool, "Loss": loss_t, "Dropout": dp, "LayerNorm": ln,
        "BioBias": bio, "HelixPE": helix, "Wimley": wim, "Moment": mom,
        "Accuracy": f"{means[0]*100:.2f}%",
        "Sensitivity": f"{means[1]*100:.2f}%",
        "Specificity": f"{means[2]*100:.2f}%",
        "MCC": f"{means[3]:.4f}",
        "AUROC": f"{means[4]:.4f}",
        "F1_Score": f"{means[5]*100:.2f}%"
    })
    print(f"  {name:<26} Acc: {means[0]*100:.2f}% | MCC: {means[3]:.4f} | AUROC: {means[4]:.4f}")

df_abl = pd.DataFrame(ablation_results)
df_abl.to_csv(os.path.join(config.RESULTS_DIR, "ablation_results.csv"), index=False)
print(f"\n  Saved -> 'results/ablation_results.csv'")

# =========================================================================
# 3. Publication Figures
# =========================================================================
print("\n[3/3] Generating Publication Figures...")

all_y_true = np.array(all_y_true)
all_y_prob = np.array(all_y_prob)
all_y_pred = (all_y_prob >= 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['non-AMP', 'AMP'], yticklabels=['non-AMP', 'AMP'])
plt.title('AMP-MHSA Confusion Matrix (5-Fold CV)', fontsize=12, fontweight='bold')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(config.FIGURES_DIR, "amp_confusion_matrix.png"), dpi=300)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
auc_val = roc_auc_score(all_y_true, all_y_prob)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AMP-MHSA (AUC = {auc_val:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve — AMP-MHSA', fontsize=12, fontweight='bold')
plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(config.FIGURES_DIR, "amp_roc_curve.png"), dpi=300)
plt.close()

# Ablation Bar Chart
abl_names = [r['Experiment'].replace('Exp', '').replace('_', '\n', 1).replace('_', ' ')
             for r in ablation_results]
abl_mccs = [float(r['MCC']) for r in ablation_results]
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(abl_mccs))]
plt.figure(figsize=(16, 6))
bars = plt.bar(range(len(abl_mccs)), abl_mccs, color=colors, edgecolor='white', linewidth=0.5)
plt.xticks(range(len(abl_mccs)), abl_names, fontsize=7, ha='center')
plt.ylabel('MCC', fontsize=11)
plt.title('12-Experiment Ablation Study — AMP-MHSA', fontsize=13, fontweight='bold')
plt.ylim(min(abl_mccs) - 0.03, max(abl_mccs) + 0.02)
for bar, val in zip(bars, abl_mccs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f'{val:.4f}',
             ha='center', va='bottom', fontsize=7, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(config.FIGURES_DIR, "ablation_comparison_chart.png"), dpi=300)
plt.close()

print(f"  Saved -> '{config.FIGURES_DIR}/'")

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\n★ Full AMP-MHSA (Lightweight Bio-Informed):")
print(f"  Accuracy   : {m_means[0]*100:.2f}% ± {m_stds[0]*100:.2f}%")
print(f"  Sensitivity: {m_means[1]*100:.2f}% ± {m_stds[1]*100:.2f}%")
print(f"  Specificity: {m_means[2]*100:.2f}% ± {m_stds[2]*100:.2f}%")
print(f"  MCC        : {m_means[3]:.4f} ± {m_stds[3]:.4f}")
print(f"  AUROC      : {m_means[4]:.4f} ± {m_stds[4]:.4f}")
print(f"  F1-Score   : {m_means[5]*100:.2f}% ± {m_stds[5]*100:.2f}%")
print(f"\n★ 12-Experiment Ablation:")
print(df_abl[['Experiment', 'Accuracy', 'MCC', 'AUROC']].to_string(index=False))
print("\n" + "=" * 70)
print("PROJECT 2 PIPELINE 100% COMPLETE!")
print("=" * 70)