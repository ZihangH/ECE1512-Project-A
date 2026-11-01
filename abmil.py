"""ABMIL baseline model (Ilse et al., ICML 2018), identical to the notebook.

Classifier1FC, DimReduction, AttentionGated and ABMIL are copied 1:1 from the completed
notebook so that the notebook and the scripted experiments use the same code path."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier1FC(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super().__init__()
        self.dropout = nn.Dropout(droprate) if droprate > 0 else nn.Identity()
        self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim):
        super().__init__()
        self.fc = nn.Linear(n_channels, m_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class AttentionGated(nn.Module):
    def __init__(self, L=256, D=128, K=1):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_weights_w = nn.Linear(D, K)

    def forward(self, x):  # x: [B, N, L]
        a_v = self.attention_V(x)
        a_u = self.attention_U(x)
        a = self.attention_weights_w(a_v * a_u)
        return a.transpose(1, 2)  # [B, K, N]


class ABMIL(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = AttentionGated(conf.D_inner, conf.D_attn, K=1)
        self.classifier = Classifier1FC(conf.D_inner, conf.n_class, conf.droprate)

    def forward(self, x, return_attention=False):
        med_feat = self.dimreduction(x)
        a_logits = self.attention(med_feat)
        attention = F.softmax(a_logits, dim=-1)
        bag_feat = torch.bmm(attention, med_feat).squeeze(1)
        logits = self.classifier(bag_feat)
        if return_attention:
            return logits, attention
        return logits
