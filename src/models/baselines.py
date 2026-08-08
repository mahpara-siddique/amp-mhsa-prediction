"""
src/models/baselines.py
========================
PyTorch implementations of Deep Learning baseline models:
1. MLP (for ESM-2 1280D mean embeddings)
2. 1D-CNN (for integer-encoded protein sequences)
3. BiLSTM (for integer-encoded protein sequences)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineMLP(nn.Module):
    """3-Layer MLP for dense ESM-2 embeddings."""
    def __init__(self, input_dim=1280, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        return self.net(x).squeeze(-1)


class Baseline1DCNN(nn.Module):
    """Multi-scale 1D CNN for integer-encoded protein sequences."""
    def __init__(self, vocab_size=21, embed_dim=128, num_filters=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Multi-scale 1D convolutions (kernel sizes 3, 5, 7)
        self.conv3 = nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, num_filters, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, num_filters, kernel_size=7, padding=3)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(num_filters * 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # x shape: [B, L]
        emb = self.embedding(x).transpose(1, 2)  # [B, embed_dim, L]
        
        c3 = F.relu(self.conv3(emb)).max(dim=-1)[0]  # [B, num_filters]
        c5 = F.relu(self.conv5(emb)).max(dim=-1)[0]  # [B, num_filters]
        c7 = F.relu(self.conv7(emb)).max(dim=-1)[0]  # [B, num_filters]
        
        concat = torch.cat([c3, c5, c7], dim=-1)     # [B, num_filters * 3]
        out = self.fc(self.dropout(concat)).squeeze(-1)
        return out


class BaselineBiLSTM(nn.Module):
    """Bidirectional LSTM with Last-State Pooling for integer-encoded sequences."""
    def __init__(self, vocab_size=21, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x shape: [B, L]
        emb = self.embedding(x)  # [B, L, embed_dim]
        out, (h_n, c_n) = self.lstm(emb)  # out shape: [B, L, hidden_dim * 2]
        
        # Max-pooling over time steps
        pooled = out.max(dim=1)[0]  # [B, hidden_dim * 2]
        logits = self.fc(pooled).squeeze(-1)
        return logits