"""Proposal A: null-option attention (NullABMIL) and its control / ablation modes.

See Sec. IV-V of the report. Modes: null, null_fixed, sigmoid (controls), null_nolog, null_val,
null_valhi, null_k4val (follow-up ablations)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from abmil import AttentionGated, Classifier1FC, DimReduction

NULL_MODES = ("null", "null_fixed", "sigmoid", "null_nolog", "null_val", "null_k4val", "null_valhi")

class NullABMIL(nn.Module):
    """ABMIL whose attention may *abstain*.

    mode = "null"       : a_i = exp(e_i) / (exp(e_0) + sum_j exp(e_j)),  e_0 = b + log N,
                          b a learnable scalar. The null instance has a ZERO value vector, so
                          z = sum_i a_i h_i  and  null mass a_0 = 1 - sum_i a_i  is free.
    mode = "null_fixed" : same, but b is frozen at 0 (control: is *learning* the null needed?)
    mode = "sigmoid"    : a_i = sigmoid(e_i) / N  (control: un-normalised attention, no softmax
                          competition at all)

    The log N offset makes the null bag-size invariant:
        a_0 = 1 / (1 + mean_j exp(e_j - b)),
    i.e. the slide abstains when the *average* attention score is below the learned level b.
    Without it a_0 ~ 1/N ~ 1e-3 at init and b (lr 1e-4) could never catch up.
    """

    def __init__(self, conf, mode="null"):
        super().__init__()
        assert mode in NULL_MODES
        self.mode = mode
        # ablation switches (Section A of the extension study)
        self.use_logn = mode != "null_nolog"            # e_0 = b + log N  vs  e_0 = b
        self.K = 4 if mode == "null_k4val" else 1       # number of null competitors
        self.learn_val = mode in ("null_val", "null_k4val", "null_valhi")  # null value vector(s) learnable
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = AttentionGated(conf.D_inner, conf.D_attn, K=1)
        self.classifier = Classifier1FC(conf.D_inner, conf.n_class, conf.droprate)
        if mode in ("null", "null_nolog", "null_val", "null_k4val", "null_valhi"):
            self.null_logit = nn.Parameter(torch.zeros(self.K))
        else:
            self.register_buffer("null_logit", torch.zeros(self.K))
        if self.learn_val:   # zero-init so training starts identical to the zero-value null
            self.null_val = nn.Parameter(torch.zeros(self.K, conf.D_inner))

    def forward(self, x, return_attention=False):
        h = self.dimreduction(x)                      # [B, N, L]
        e = self.attention(h)                         # [B, 1, N]
        N = e.shape[-1]
        if self.mode == "sigmoid":
            a = torch.sigmoid(e) / N
        else:
            off = math.log(N) if self.use_logn else 0.0
            e0 = (self.null_logit + off).view(1, 1, self.K).expand(e.shape[0], 1, self.K)
            full = F.softmax(torch.cat([e0, e], dim=-1), dim=-1)   # [B, 1, K+N]
            a = full[..., self.K:]
        z = torch.bmm(a, h).squeeze(1)
        if self.learn_val:   # null mass is routed to a learned "default" embedding
            z = z + full[..., :self.K].squeeze(1) @ self.null_val
        logits = self.classifier(z)
        if return_attention:
            return logits, a, a.sum(dim=-1).squeeze(1)   # evidence = sum_i a_i = 1 - null mass
        return logits
