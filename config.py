"""
config.py
=========
Global configuration, paths, random seeds, and hyperparameters for Project 2.
"""
import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "features")

FOLDS_DIR = os.path.join(BASE_DIR, "folds")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# Reproducibility
SEED = 42

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ESM-2 Configuration
ESM_MODEL_NAME = "esm2_t33_650M_UR50D"
ESM_EMBED_DIM = 1280