"""
src/features.py
===============
Feature calculation algorithms:
1. Amino Acid Composition (AAC - 20D)
2. Dipeptide Composition (DPC - 400D)
3. Physicochemical Properties (10D)
4. Combined Classical Features (430D)
"""

import numpy as np
from collections import Counter
from itertools import product

STANDARD_AA = sorted(list("ACDEFGHIKLMNPQRSTVWY"))

# Physicochemical scales (Charge, Hydrophobicity, pI, Molecular Weight, Isoelectric Point)
HYDROPHOBICITY = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2,
                  'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5,
                  'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}

CHARGE = {'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0, 'G': 0, 'H': 0.1, 'I': 0, 'K': 1,
          'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}

MOLECULAR_WEIGHT = {'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
                    'G': 75.07, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
                    'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
                    'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19}

def extract_aac(sequences):
    """Amino Acid Composition (20D)."""
    features = []
    for seq in sequences:
        total = max(len(seq), 1)
        counts = Counter(seq)
        features.append([counts.get(aa, 0) / total for aa in STANDARD_AA])
    return np.array(features, dtype=np.float32)

def extract_dpc(sequences):
    """Dipeptide Composition (400D)."""
    dipeptides = [''.join(dp) for dp in product(STANDARD_AA, repeat=2)]
    features = []
    for seq in sequences:
        total = max(len(seq) - 1, 1)
        counts = Counter(seq[i:i+2] for i in range(len(seq) - 1))
        features.append([counts.get(dp, 0) / total for dp in dipeptides])
    return np.array(features, dtype=np.float32)

def extract_physicochemical(sequences):
    """Physicochemical properties (10D)."""
    features = []
    for seq in sequences:
        l = max(len(seq), 1)
        mean_hydro = sum(HYDROPHOBICITY.get(aa, 0) for aa in seq) / l
        mean_charge = sum(CHARGE.get(aa, 0) for aa in seq)
        mean_mw = sum(MOLECULAR_WEIGHT.get(aa, 0) for aa in seq) / l
        pos_charge = sum(1 for aa in seq if CHARGE.get(aa, 0) > 0) / l
        neg_charge = sum(1 for aa in seq if CHARGE.get(aa, 0) < 0) / l
        aromatic = sum(1 for aa in seq if aa in "FYW") / l
        aliphatic = sum(1 for aa in seq if aa in "AIVL") / l
        polar = sum(1 for aa in seq if aa in "NCQST") / l
        tiny = sum(1 for aa in seq if aa in "GAS") / l
        proline_ratio = seq.count('P') / l
        
        features.append([mean_hydro, mean_charge, mean_mw, pos_charge, neg_charge,
                         aromatic, aliphatic, polar, tiny, proline_ratio])
    return np.array(features, dtype=np.float32)

def extract_all_classical(sequences):
    """Combined Classical Features (430D)."""
    aac = extract_aac(sequences)
    dpc = extract_dpc(sequences)
    phys = extract_physicochemical(sequences)
    return np.hstack([aac, dpc, phys])