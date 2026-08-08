"""
scripts/03_train_baselines.py
=============================
Phase 2 - Steps 4, 5 & 6: 
Train 9 Baseline Models across 5-Fold Stratified Cross-Validation.

Metrics reported per fold: Accuracy, Sensitivity (Recall), Specificity, MCC, AUROC, F1-Score.
Saves summary to 'results/baseline_cv_results.csv'.
"""

import os
import sys

# Ensure root project directory is in Python path FIRST
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Always import sklearn before torch on Windows
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, matthews_corrcoef, roc_auc_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import config
from src.models.baselines import BaselineMLP, Baseline1DCNN, BaselineBiLSTM

# Set Random Seed
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

print("=" * 70)
print("PHASE 2 - STEPS 4-6: BASELINE MODELS & 5-FOLD CV BENCHMARKING")
print("=" * 70)

# Load Features and Master Dataset
X_classical = np.load(os.path.join(config.FEATURES_DIR, "classical_430d.npy"))
X_esm2 = np.load(os.path.join(config.FEATURES_DIR, "esm2_mean_1280d.npy"))
labels = np.load(os.path.join(config.FEATURES_DIR, "labels.npy"))

MASTER_CSV = os.path.join(config.PROCESSED_DIR, "amp_benchmark_3556.csv")
df_master = pd.read_csv(MASTER_CSV)
sequences = df_master['sequence'].tolist()
folds_array = df_master['fold'].values  # Fold column (0..4)

# Integer encoding mapping for sequence-based DL models (CNN, BiLSTM)
STANDARD_AA = sorted(list("ACDEFGHIKLMNPQRSTVWY"))
AA_TO_INT = {aa: idx + 1 for idx, aa in enumerate(STANDARD_AA)}  # 0 is padding

def encode_sequence(seq, max_len=100):
    encoded = [AA_TO_INT.get(aa, 0) for aa in seq[:max_len]]
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    return encoded

X_seq_encoded = np.array([encode_sequence(s) for s in sequences], dtype=np.int64)

# Calculate Evaluation Metrics
def calc_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)  # Sensitivity / Recall
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
        
    return acc, sens, spec, mcc, auc, f1

# -------------------------------------------------------------------------
# Define All 9 Baseline Models
# -------------------------------------------------------------------------
ml_models = {
    "Naive Bayes (Classical 430D)": (GaussianNB(), "classical"),
    "KNN k=5 (Classical 430D)": (KNeighborsClassifier(n_neighbors=5), "classical"),
    "Logistic Regression (ESM-2)": (LogisticRegression(max_iter=1000, random_state=config.SEED), "esm2"),
    "Random Forest (ESM-2)": (RandomForestClassifier(n_estimators=200, random_state=config.SEED), "esm2"),
    "XGBoost (ESM-2)": (XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=config.SEED), "esm2"),
    "SVM RBF (ESM-2)": (SVC(kernel="rbf", probability=True, random_state=config.SEED), "esm2"),
}

results_list = []

# -------------------------------------------------------------------------
# 1. Train & Evaluate 6 ML Baselines over 5 Folds
# -------------------------------------------------------------------------
print("\n[1/2] Evaluating 6 Machine Learning Baseline Models across 5 Folds...")

for model_name, (clf, feat_type) in ml_models.items():
    print(f"\n---> Training: {model_name}")
    X_data = X_classical if feat_type == "classical" else X_esm2
    
    fold_metrics = []
    
    for f in range(5):
        val_indices = np.where(folds_array == f)[0]
        train_indices = np.where(folds_array != f)[0]
        
        X_train, y_train = X_data[train_indices], labels[train_indices]
        X_val, y_val = X_data[val_indices], labels[val_indices]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]
        
        metrics = calc_metrics(y_val, y_pred, y_prob)
        fold_metrics.append(metrics)
        
    fold_metrics = np.array(fold_metrics)
    means = fold_metrics.mean(axis=0)
    stds = fold_metrics.std(axis=0)
    
    results_list.append({
        "Model": model_name,
        "Feature_Type": feat_type,
        "Accuracy": f"{means[0]*100:.2f}% ± {stds[0]*100:.2f}%",
        "Sensitivity": f"{means[1]*100:.2f}% ± {stds[1]*100:.2f}%",
        "Specificity": f"{means[2]*100:.2f}% ± {stds[2]*100:.2f}%",
        "MCC": f"{means[3]:.4f} ± {stds[3]:.4f}",
        "AUROC": f"{means[4]:.4f} ± {stds[4]:.4f}",
        "F1_Score": f"{means[5]*100:.2f}% ± {stds[5]*100:.2f}%",
        "Raw_Acc_Mean": means[0],
        "Raw_MCC_Mean": means[3]
    })
    print(f"  Result: Acc = {means[0]*100:.2f}% | MCC = {means[3]:.4f} | AUROC = {means[4]:.4f}")

# -------------------------------------------------------------------------
# 2. Train & Evaluate 3 PyTorch Deep Learning Baselines over 5 Folds
# -------------------------------------------------------------------------
print("\n[2/2] Evaluating 3 PyTorch Deep Learning Baseline Models across 5 Folds...")

dl_models = [
    ("MLP (ESM-2 1280D)", "mlp", X_esm2),
    ("1D-CNN (Sequences)", "cnn", X_seq_encoded),
    ("BiLSTM (Sequences)", "bilstm", X_seq_encoded),
]

def train_pytorch_model(model_type, X_tr, y_tr, X_va, y_va, epochs=25):
    if model_type == "mlp":
        net = BaselineMLP(input_dim=1280).to(config.DEVICE)
    elif model_type == "cnn":
        net = Baseline1DCNN().to(config.DEVICE)
    elif model_type == "bilstm":
        net = BaselineBiLSTM().to(config.DEVICE)
        
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    
    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32 if model_type == "mlp" else torch.long), torch.tensor(y_tr, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        net.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(config.DEVICE), batch_y.to(config.DEVICE)
            optimizer.zero_grad()
            logits = net(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
    net.eval()
    with torch.no_grad():
        val_x = torch.tensor(X_va, dtype=torch.float32 if model_type == "mlp" else torch.long).to(config.DEVICE)
        val_logits = net(val_x)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_preds = (val_probs >= 0.5).astype(int)
        
    return val_preds, val_probs

for model_name, model_type, X_data in dl_models:
    print(f"\n---> Training: {model_name}")
    fold_metrics = []
    
    for f in range(5):
        val_indices = np.where(folds_array == f)[0]
        train_indices = np.where(folds_array != f)[0]
        
        X_train, y_train = X_data[train_indices], labels[train_indices]
        X_val, y_val = X_data[val_indices], labels[val_indices]
        
        y_pred, y_prob = train_pytorch_model(model_type, X_train, y_train, X_val, y_val, epochs=25)
        metrics = calc_metrics(y_val, y_pred, y_prob)
        fold_metrics.append(metrics)
        
    fold_metrics = np.array(fold_metrics)
    means = fold_metrics.mean(axis=0)
    stds = fold_metrics.std(axis=0)
    
    results_list.append({
        "Model": model_name,
        "Feature_Type": "esm2" if model_type == "mlp" else "sequence",
        "Accuracy": f"{means[0]*100:.2f}% ± {stds[0]*100:.2f}%",
        "Sensitivity": f"{means[1]*100:.2f}% ± {stds[1]*100:.2f}%",
        "Specificity": f"{means[2]*100:.2f}% ± {stds[2]*100:.2f}%",
        "MCC": f"{means[3]:.4f} ± {stds[3]:.4f}",
        "AUROC": f"{means[4]:.4f} ± {stds[4]:.4f}",
        "F1_Score": f"{means[5]*100:.2f}% ± {stds[5]*100:.2f}%",
        "Raw_Acc_Mean": means[0],
        "Raw_MCC_Mean": means[3]
    })
    print(f"  Result: Acc = {means[0]*100:.2f}% | MCC = {means[3]:.4f} | AUROC = {means[4]:.4f}")

# -------------------------------------------------------------------------
# Save & Rank Results (Step 6: Baseline Selection)
# -------------------------------------------------------------------------
df_results = pd.DataFrame(results_list)
df_results = df_results.sort_values(by="Raw_MCC_Mean", ascending=False).reset_index(drop=True)

# Drop raw sorting columns for clean export
df_clean_export = df_results.drop(columns=["Raw_Acc_Mean", "Raw_MCC_Mean"])

CSV_OUT = os.path.join(config.RESULTS_DIR, "baseline_cv_results.csv")
df_clean_export.to_csv(CSV_OUT, index=False)

print("\n" + "=" * 70)
print("PHASE 2 SUMMARY & BASELINE SELECTION (STEP 6)")
print("=" * 70)
print(df_clean_export.to_string(index=False))

best_model = df_clean_export.iloc[0]["Model"]
best_mcc = df_clean_export.iloc[0]["MCC"]
best_acc = df_clean_export.iloc[0]["Accuracy"]

print(f"\n🏆 BEST PERFORMING BASELINE MODEL: {best_model}")
print(f"   Accuracy: {best_acc} | MCC: {best_mcc}")
print(f"\nSaved Baseline Comparison Table -> '{CSV_OUT}'")
print("=" * 70)
print("PHASE 2 COMPLETE: READY FOR NOVEL ARCHITECTURE DISCOVERY (STEPS 7-12)!")
print("=" * 70)