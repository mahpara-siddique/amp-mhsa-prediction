"""
src/models/proposed_model.py
=============================
AMP-MHSA: Antimicrobial Peptide Multi-Head Self-Attention Network

Lightweight Biology-Informed Architecture (~3,000 extra params vs ~2.5M before):
1. Bio-Informed Attention Bias: PhysChem features → scalar importance per residue
   → additive bias in attention scores (~449 params)
2. Amphipathic Helical Positional Encoding: period=3.6 residues (~2,560 params)
3. Hydrophobic Moment Feature: Eisenberg formula μH per residue (0 model params)
4. Wimley-White Membrane Transfer Energy per residue (0 model params)
5. Dual-Path Pooling: Bio-biased attention + global mean → concat [2560D]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# Biophysical Lookup Tables
# =========================================================================

# Kyte-Doolittle Hydrophobicity Scale
_HYDRO = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2,
          'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5,
          'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}

# Formal Charge at Physiological pH
_CHARGE = {'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0, 'G': 0, 'H': 0.1, 'I': 0,
           'K': 1, 'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1, 'S': 0,
           'T': 0, 'V': 0, 'W': 0, 'Y': 0}

# Molecular Weight (Daltons)
_MW = {'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
       'G': 75.07, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
       'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
       'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19}

# Wimley-White Whole-Residue Free Energies: Water -> POPC Interface (kcal/mol)
# Source: Wimley & White, Nature Struct. Biol. 3, 842-848 (1996)
_WIMLEY_WHITE = {
    'A':  0.17, 'C': -0.24, 'D':  1.23, 'E':  2.02, 'F': -1.13,
    'G':  0.01, 'H':  0.96, 'I': -0.31, 'K':  0.99, 'L': -0.56,
    'M': -0.23, 'N':  0.42, 'P':  0.45, 'Q':  0.58, 'R':  0.81,
    'S':  0.13, 'T':  0.14, 'V':  0.07, 'W': -1.85, 'Y': -0.94
}


def compute_residue_physicochemical(sequence):
    """
    11 physicochemical properties per residue -> [L, 11].
    [0]  Kyte-Doolittle Hydrophobicity (norm)  [1] Charge  [2] MW (norm)
    [3]  Cationic  [4] Anionic  [5] Aromatic  [6] Aliphatic
    [7]  Polar     [8] Tiny     [9] Proline   [10] Wimley-White ΔG (norm)
    """
    feats = []
    for aa in sequence:
        feats.append([
            (_HYDRO.get(aa, 0) + 4.5) / 9.0,
            _CHARGE.get(aa, 0),
            (_MW.get(aa, 130) - 75) / 130.0,
            1.0 if aa in "KRH" else 0.0,
            1.0 if aa in "DE" else 0.0,
            1.0 if aa in "FYW" else 0.0,
            1.0 if aa in "AIVL" else 0.0,
            1.0 if aa in "NCQST" else 0.0,
            1.0 if aa in "GAS" else 0.0,
            1.0 if aa == "P" else 0.0,
            (_WIMLEY_WHITE.get(aa, 0) + 1.85) / 3.87,
        ])
    return torch.tensor(feats, dtype=torch.float32)


def compute_hydrophobic_moment(sequence, window=11):
    """
    Per-position hydrophobic moment (Eisenberg formula, sliding window).
    δ = 100° (angular offset per residue in α-helix).
    Returns list of L floats, normalized to [0, 1].
    """
    delta = 100.0 * math.pi / 180.0
    raw_hydro = [_HYDRO.get(aa, 0.0) for aa in sequence]
    L = len(sequence)
    moments = []
    half_w = window // 2
    for i in range(L):
        start = max(0, i - half_w)
        end = min(L, i + half_w + 1)
        sin_sum, cos_sum = 0.0, 0.0
        for j, pos in enumerate(range(start, end)):
            sin_sum += raw_hydro[pos] * math.sin(j * delta)
            cos_sum += raw_hydro[pos] * math.cos(j * delta)
        moments.append(math.sqrt(sin_sum ** 2 + cos_sum ** 2))
    max_m = max(moments) if moments and max(moments) > 0 else 1.0
    return [m / max_m for m in moments]


def compute_all_residue_features(sequence):
    """
    Complete per-residue feature vector [L, 12]:
    [0-10]: 11 PhysChem features (incl. Wimley-White)
    [11]:   Local Hydrophobic Moment (Eisenberg, window=11)
    """
    physchem = compute_residue_physicochemical(sequence)        # [L, 11]
    moments = compute_hydrophobic_moment(sequence)               # list[L]
    moment_t = torch.tensor(moments, dtype=torch.float32).unsqueeze(-1)  # [L, 1]
    return torch.cat([physchem, moment_t], dim=-1)               # [L, 12]


# =========================================================================
# Lightweight Novel Components
# =========================================================================

class BioImportanceScorer(nn.Module):
    """
    Ultra-lightweight MLP: 12D PhysChem -> scalar importance per residue.
    Added as attention bias to nudge attention toward biologically important
    residues (cationic, amphipathic, membrane-interacting).

    Total parameters: ~449  (vs ~2.5M in the previous cross-attention design)
    """
    def __init__(self, phys_dim=12):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(phys_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, physchem, mask=None):
        scores = self.scorer(physchem).squeeze(-1)  # [B, L]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, 0.0)
        return self.scale * scores


class HelicalPositionalEncoding(nn.Module):
    """
    Sinusoidal PE with period = 3.6 residues (one α-helix turn).
    Projected from 2D -> embed_dim and added to ESM-2 embeddings.
    Total parameters: ~2,560
    """
    def __init__(self, embed_dim=1280):
        super().__init__()
        self.proj = nn.Linear(2, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        angle = 2.0 * math.pi * positions / 3.6
        pe_2d = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        return self.proj(pe_2d).unsqueeze(0)  # [1, L, embed_dim]


class SelfAttentionPooling(nn.Module):
    """
    Standard multi-head self-attention with learnable query.
    Supports additive bio_bias from BioImportanceScorer.
    """
    def __init__(self, embed_dim=1280, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Parameter(torch.randn(num_heads, 1, self.head_dim))
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.val_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.query)

    def forward(self, x, mask=None, bio_bias=None):
        B, L, D = x.shape
        K = self.key_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.val_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        Q = self.query.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, 1, d_k]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, 1, L]

        # Novel: Additive biological importance bias
        if bio_bias is not None:
            scores = scores + bio_bias.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]

        if mask is not None:
            mask_4d = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_4d == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # [B, H, 1, L]
        context = torch.matmul(attn_weights, V)     # [B, H, 1, d_k]
        pooled = context.view(B, D)                  # [B, 1280]

        attn_viz = attn_weights.squeeze(2).mean(dim=1)  # [B, L]
        return pooled, attn_viz


# =========================================================================
# Complete AMP-MHSA Architecture
# =========================================================================

class AMP_MHSA(nn.Module):
    """
    AMP-MHSA: Lightweight Bio-Informed Architecture.

    Ablation flags:
        use_bio_bias:  Bio importance scorer as attention bias (~449 params)
        use_helix_pe:  Helical positional encoding (~2,560 params)
        use_wimley:    Wimley-White membrane feature (0 model params)
        use_moment:    Hydrophobic moment feature (0 model params)
        pooling_mode:  'fused' (concat), 'attn_only', 'mean_only'
    """
    def __init__(self, embed_dim=1280, phys_dim=12, num_heads=4,
                 pooling_mode="fused", dropout=0.4, use_layernorm=True,
                 use_bio_bias=True, use_helix_pe=True,
                 use_wimley=True, use_moment=True):
        super().__init__()
        self.pooling_mode = pooling_mode
        self.use_layernorm = use_layernorm
        self.use_bio_bias = use_bio_bias
        self.use_helix_pe = use_helix_pe
        self.use_wimley = use_wimley
        self.use_moment = use_moment

        if use_helix_pe:
            self.helix_pe = HelicalPositionalEncoding(embed_dim=embed_dim)

        if use_bio_bias:
            self.bio_scorer = BioImportanceScorer(phys_dim=phys_dim)

        self.self_attn = SelfAttentionPooling(embed_dim=embed_dim, num_heads=num_heads)

        fc_input_dim = embed_dim * 2 if pooling_mode == "fused" else embed_dim

        if use_layernorm:
            self.layer_norm = nn.LayerNorm(fc_input_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, esm_emb, physchem=None, mask=None):
        B, L, D = esm_emb.shape
        if mask is None:
            mask = torch.ones(B, L, device=esm_emb.device)

        # Zero out disabled features
        if physchem is not None and (not self.use_wimley or not self.use_moment):
            physchem = physchem.clone()
            if not self.use_wimley:
                physchem[:, :, 10] = 0.0
            if not self.use_moment:
                physchem[:, :, 11] = 0.0

        # Helical Positional Encoding
        if self.use_helix_pe:
            esm_emb = esm_emb + self.helix_pe(L, esm_emb.device)

        # Bio Importance Bias
        bio_bias = None
        if self.use_bio_bias and physchem is not None:
            bio_bias = self.bio_scorer(physchem, mask)

        # Path A: Self-Attention Pooling (with bio bias)
        attn_pooled, attn_weights = self.self_attn(esm_emb, mask, bio_bias)

        # Path B: Global Mean Pooling
        mask_exp = mask.unsqueeze(-1)
        mean_pooled = (esm_emb * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)

        # Fusion
        if self.pooling_mode == "fused":
            fused = torch.cat([attn_pooled, mean_pooled], dim=-1)
        elif self.pooling_mode == "attn_only":
            fused = attn_pooled
        else:
            fused = mean_pooled

        if self.use_layernorm:
            fused = self.layer_norm(fused)

        logits = self.classifier(fused).squeeze(-1)
        return logits, attn_weights


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        return ((1 - pt) ** self.gamma * bce).mean()