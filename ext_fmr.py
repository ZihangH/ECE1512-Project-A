"""Proposal B: Focal-Mass Readout (FocalMassABMIL) and its ablation variants (fmr, shuffle, total).

See the FMR appendix of the report."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from abmil import AttentionGated, Classifier1FC, DimReduction

FMR_RADII = (1, 4, 16)   # Chebyshev radii in 512-px grid cells (3x3, 9x9, 33x33 windows)


class FocalMassABMIL(nn.Module):
    """ABMIL + Focal-Mass Readout.

    A per-patch 'tumour-likeness' score s_i = sigmoid(w.h_i + b) is computed from the reduced
    features. s is scattered onto the slide's 512-px grid and box-summed within Chebyshev radius
    r (r in FMR_RADII) giving the neighbourhood mass m_i^r. The readout feature for scale r is
    log(1 + max_i m_i^r): "how much suspicious tissue is packed into the densest focus of scale r".
    These F features are concatenated with the ABMIL bag embedding before the classifier.

    variant:
      'fmr'     : true patch positions
      'shuffle' : same computation, but patch->grid-cell assignment is randomly permuted
                  (controls for 'a max over smoothed scores' without real geometry)
      'total'   : single feature log(1 + sum_i s_i)  (no spatial information at all)
    forward(x, coords) where coords is a [N,2] long tensor of grid cells (row, col).
    """

    def __init__(self, conf, variant="fmr", radii=FMR_RADII):
        super().__init__()
        assert variant in ("fmr", "shuffle", "total")
        self.variant, self.radii = variant, tuple(radii)
        self.n_extra = 1 if variant == "total" else len(self.radii)
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = AttentionGated(conf.D_inner, conf.D_attn, K=1)
        self.score = nn.Linear(conf.D_inner, 1)
        self.classifier = Classifier1FC(conf.D_inner + self.n_extra, conf.n_class, conf.droprate)
        self.permute_coords_at_test = False   # diagnostic switch for the 'fmr' variant
        self.seed_eval = 0

    def _maybe_permute(self, coords):
        n = coords.shape[0]
        if self.variant == "shuffle" or (self.variant == "fmr" and self.permute_coords_at_test):
            if self.training:
                perm = torch.randperm(n, device=coords.device)
            else:
                g = torch.Generator().manual_seed(self.seed_eval)
                perm = torch.randperm(n, generator=g).to(coords.device)
            return coords[perm]
        return coords

    def focal_features(self, s, coords):
        """s: [N] scores in (0,1); coords: [N,2] long -> [n_extra] features."""
        if self.variant == "total":
            return torch.log1p(s.sum()).view(1)
        coords = self._maybe_permute(coords)
        H, W = int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1
        R = max(self.radii)
        grid = torch.zeros(H * W, device=s.device, dtype=s.dtype)
        grid = grid.index_put((coords[:, 0] * W + coords[:, 1],), s, accumulate=True).view(1, 1, H, W)
        feats = []
        for r in self.radii:
            k = 2 * r + 1
            mass = F.avg_pool2d(grid, k, stride=1, padding=r, count_include_pad=True) * (k * k)
            m = mass[0, 0, coords[:, 0], coords[:, 1]]          # neighbourhood mass at each patch
            feats.append(torch.log1p(m.max()))
        return torch.stack(feats)

    def forward(self, x, coords, return_extra=False):
        assert x.shape[0] == 1
        h = self.dimreduction(x)                                  # [1,N,L]
        att = F.softmax(self.attention(h), dim=-1)
        z = torch.bmm(att, h).squeeze(1)                          # [1,L]
        s = torch.sigmoid(self.score(h)).view(-1)                 # [N]
        f = self.focal_features(s, coords).view(1, -1)            # [1,F]
        logits = self.classifier(torch.cat([z, f], dim=1))
        return (logits, f, s) if return_extra else logits
