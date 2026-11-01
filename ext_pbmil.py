"""Two-tier pseudo-bag ABMIL (DTFD-MIL-style, Zhang et al., CVPR 2022), re-implemented from scratch.

Exploratory extension reported in the appendix."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from abmil import AttentionGated, Classifier1FC, DimReduction

class PseudoBagABMIL(nn.Module):
    """Tier-1: shared ABMIL applied to M random pseudo-bags of one slide.
    Tier-2: second ABMIL over the M (detached) pseudo-bag embeddings.

    forward(x) returns a dict with
        'logits'  : slide-level logits from tier 2             [B, C]
        'logits1' : per-pseudo-bag logits from tier 1          [M, C]  (training only)
    """

    def __init__(self, conf, num_pseudo_bags=5, detach_tier2=True, seed_eval=0):
        super().__init__()
        self.M = int(num_pseudo_bags)
        self.detach_tier2 = detach_tier2
        self.seed_eval = seed_eval
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention1 = AttentionGated(conf.D_inner, conf.D_attn, K=1)
        self.classifier1 = Classifier1FC(conf.D_inner, conf.n_class, conf.droprate)
        self.attention2 = AttentionGated(conf.D_inner, conf.D_attn, K=1)
        self.classifier2 = Classifier1FC(conf.D_inner, conf.n_class, conf.droprate)

    def _split(self, n, device):
        m = min(self.M, n)
        if self.training:
            perm = torch.randperm(n, device=device)
        else:  # fixed, seeded split -> deterministic predictions at test time
            g = torch.Generator().manual_seed(self.seed_eval)
            perm = torch.randperm(n, generator=g).to(device)
        return torch.chunk(perm, m)

    def forward(self, x):
        assert x.shape[0] == 1, "PseudoBagABMIL assumes batch size 1 (as in the baseline)"
        feats = self.dimreduction(x)  # [1, N, L]
        chunks = self._split(feats.shape[1], feats.device)
        bag_feats, logits1 = [], []
        for idx in chunks:
            h = feats[:, idx]                                   # [1, n_m, L]
            a = F.softmax(self.attention1(h), dim=-1)           # [1, 1, n_m]
            z = torch.bmm(a, h).squeeze(1)                      # [1, L]
            bag_feats.append(z)
            logits1.append(self.classifier1(z))
        z_all = torch.stack(bag_feats, dim=1)                   # [1, M, L]
        if self.detach_tier2:
            z_all = z_all.detach()
        a2 = F.softmax(self.attention2(z_all), dim=-1)          # [1, 1, M]
        slide_feat = torch.bmm(a2, z_all).squeeze(1)            # [1, L]
        return {"logits": self.classifier2(slide_feat), "logits1": torch.cat(logits1, 0)}

    @torch.no_grad()
    def explain(self, x):
        """Slide-level logits and a per-patch attention map (original patch order).
        attention_i = alpha2[m(i)] * alpha1[i]: tier-2 weight of the pseudo-bag that
        contains patch i times its tier-1 weight inside that pseudo-bag (sums to 1)."""
        assert x.shape[0] == 1
        feats = self.dimreduction(x)
        chunks = self._split(feats.shape[1], feats.device)
        zs, a1s = [], []
        for idx in chunks:
            h = feats[:, idx]
            a = F.softmax(self.attention1(h), dim=-1)
            zs.append(torch.bmm(a, h).squeeze(1)); a1s.append(a.reshape(-1))
        z_all = torch.stack(zs, dim=1)
        a2 = F.softmax(self.attention2(z_all), dim=-1).reshape(-1)
        slide_feat = torch.bmm(a2.view(1, 1, -1), z_all).squeeze(1)
        att = torch.zeros(feats.shape[1], device=feats.device)
        for m, idx in enumerate(chunks):
            att[idx] = a2[m] * a1s[m]
        return self.classifier2(slide_feat), att.view(1, 1, -1)

    def forward_with_attention(self, x):
        logits, att = self.explain(x)
        return logits, att
