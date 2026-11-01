"""Shared MIL engine for ECE1512 Project A: cached data loading, train / evaluate loops, model factory.

The baseline model lives in abmil.py. Extensions live in ext_null.py (Proposal A, null-option
attention), ext_fmr.py (Proposal B, focal-mass readout) and ext_pbmil.py (pseudo-bag appendix);
build_model() imports them lazily.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from abmil import ABMIL
from dataloader import build_HDF5_feat_dataset, mil_collate_fn
from utils import classification_metrics, infer_feature_dim, load_config, seed_everything

CONFIGS = {
    "C16": "config/camelyon16_medical_ssl_config.yml",
    "C17": "config/camelyon17_medical_ssl_config.yml",
    "BRACS": "config/bracs_medical_ssl_config.yml",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
PATCH_STRIDE = 512   # median nearest-neighbour distance between patch centres (level-0 px)


def grid_cells(coords):
    """Bin level-0 pixel coordinates [N,2] into 512-px grid cells (row, col), origin at the min."""
    if coords is None:
        return None
    c = coords.long()
    c = c - c.min(0).values
    return torch.div(c, PATCH_STRIDE, rounding_mode="floor")


class CachedDataset(Dataset):
    """Reads every slide from the HDF5 file once and keeps it in RAM as float16.
    Converted to float32 on access (same values as the lazy loader)."""

    def __init__(self, base):
        self.names = list(base.data_names)
        self.items = []
        for i in range(len(base)):
            d = base[i]
            self.items.append((d["input"].half(), d["label"], d["slide_id"], grid_cells(d["coords"])))
        self.feature_dim = self.items[0][0].shape[-1]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y, s, c = self.items[i]
        return {"input": x.float(), "coords": c, "label": y, "slide_id": s}


def load_data(dataset, project_dir=".", cache=True):
    conf = load_config(os.path.join(project_dir, CONFIGS[dataset]))
    conf.data_dir = os.path.join(project_dir, conf.data_dir)
    conf.dataset_csv_dir = os.path.join(project_dir, conf.dataset_csv_dir)
    tr, va, te = build_HDF5_feat_dataset(conf.data_dir, conf)
    conf.D_feat = infer_feature_dim(tr)
    if cache:
        tr, va, te = CachedDataset(tr), CachedDataset(va), CachedDataset(te)
    return conf, tr, va, te


def make_loaders(tr, va, te, seed):
    g = torch.Generator().manual_seed(seed)
    kw = dict(batch_size=1, num_workers=0, collate_fn=mil_collate_fn)
    return (DataLoader(tr, shuffle=True, generator=g, **kw),
            DataLoader(va, shuffle=False, **kw),
            DataLoader(te, shuffle=False, **kw))


# --------------------------------------------------------------------------- #
# Train / eval (same logic as the notebook; handles every method)
# --------------------------------------------------------------------------- #
# Every method returns plain logits except the two-tier pseudo-bag model ("pbmil").


def train_one_epoch(model, criterion, loader, optimizer, device, method="abmil"):
    model.train()
    tot_loss = tot_correct = tot_n = 0
    for data in loader:
        bag = data["input"].to(device, dtype=torch.float32)
        labels = data["label"].to(device, dtype=torch.long)
        optimizer.zero_grad()
        out = model(bag, data["coords"][0].to(device)) if method.startswith("fmr") else model(bag)
        if method != "pbmil":
            logits = out
            loss = criterion(logits, labels)
        else:
            logits = out["logits"]
            l1 = criterion(out["logits1"], labels.expand(out["logits1"].shape[0]))
            loss = l1 + criterion(logits, labels)
        loss.backward()
        optimizer.step()
        n = labels.shape[0]
        tot_loss += loss.item() * n
        tot_correct += (logits.argmax(1) == labels).sum().item()
        tot_n += n
    return {"loss": tot_loss / tot_n, "acc": 100.0 * tot_correct / tot_n}


@torch.no_grad()
def evaluate(model, criterion, loader, device, conf, method="abmil"):
    model.eval()
    tot_loss = tot_n = 0
    probs_all, labels_all = [], []
    for data in loader:
        bag = data["input"].to(device, dtype=torch.float32)
        labels = data["label"].to(device, dtype=torch.long)
        out = model(bag, data["coords"][0].to(device)) if method.startswith("fmr") else model(bag)
        logits = out if method != "pbmil" else out["logits"]
        loss = criterion(logits, labels)
        tot_loss += loss.item() * labels.shape[0]
        tot_n += labels.shape[0]
        probs_all.append(F.softmax(logits, dim=1).cpu())
        labels_all.append(labels.cpu())
    m = classification_metrics(torch.cat(probs_all), torch.cat(labels_all), conf.n_class)
    m["loss"] = tot_loss / tot_n
    return m


def build_model(conf, method, M=5, detach=True):
    """Model factory. Extension modules are imported lazily so the baseline never depends on them."""
    if method == "abmil":
        return ABMIL(conf)
    if method.startswith("null") or method == "sigmoid":      # Proposal A (ext_null.py)
        from ext_null import NULL_MODES, NullABMIL
        if method in NULL_MODES:
            return NullABMIL(conf, method)
    if method == "pbmil":                                      # appendix (ext_pbmil.py)
        from ext_pbmil import PseudoBagABMIL
        return PseudoBagABMIL(conf, num_pseudo_bags=M, detach_tier2=detach)
    if method.startswith("fmr"):                               # Proposal B (ext_fmr.py)
        from ext_fmr import FocalMassABMIL
        variant = {"fmr": "fmr", "fmr_shuffle": "shuffle", "fmr_total": "total"}.get(method)
        if variant:
            return FocalMassABMIL(conf, variant)
    raise ValueError(method)


def run_one(conf, tr, va, te, method="abmil", seed=42, epochs=None, device="cpu",
            M=5, detach=True, log=print):
    """Train one model; select the epoch on validation (auc+f1); test once."""
    epochs = int(epochs or conf.train_epoch)
    seed_everything(seed)
    device = torch.device(device)
    tl, vl, sl = make_loaders(tr, va, te, seed)
    model = build_model(conf, method, M, detach).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss()
    if method == "null_valhi":   # null value vector gets a 100x larger lr (default lr is too small to move it)
        hi = [p for n, p in model.named_parameters() if n == "null_val"]
        lo = [p for n, p in model.named_parameters() if n != "null_val"]
        opt = torch.optim.AdamW([{"params": lo}, {"params": hi, "lr": 100 * float(conf.lr)}],
                                lr=float(conf.lr), weight_decay=float(conf.wd))
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=float(conf.lr), weight_decay=float(conf.wd))
    best_score, best_epoch, best_state, hist = float("-inf"), -1, None, []
    for ep in range(epochs):
        t0 = time.time()
        trm = train_one_epoch(model, criterion, tl, opt, device, method)
        vam = evaluate(model, criterion, vl, device, conf, method)
        score = vam["auc"] + vam["f1"]
        if score > best_score:
            best_score, best_epoch, best_state = score, ep, copy.deepcopy(model.state_dict())
        hist.append({"epoch": ep, "train_loss": trm["loss"], "train_acc": trm["acc"],
                     "val_loss": vam["loss"], "val_acc": vam["acc"],
                     "val_auc": vam["auc"], "val_f1": vam["f1"]})
        log(f"[{method} s{seed}] ep {ep+1:02d}/{epochs} train loss={trm['loss']:.4f} "
            f"acc={trm['acc']:.1f} | val loss={vam['loss']:.4f} auc={vam['auc']:.3f} "
            f"f1={vam['f1']:.3f} | best={best_epoch+1} | {time.time()-t0:.1f}s")
    model.load_state_dict(best_state)
    test = evaluate(model, criterion, sl, device, conf, method)
    log(f"[{method} s{seed}] TEST (best epoch {best_epoch+1}): " +
        " ".join(f"{k}={v:.4f}" for k, v in test.items()))
    extra = ({"null_logit": float(model.null_logit.item()) if model.null_logit.numel() == 1
              else model.null_logit.detach().flatten().tolist()} if hasattr(model, "null_logit") else {})
    return {**extra, "method": method, "seed": seed, "M": M, "detach": detach, "n_params": n_params,
            "best_epoch": best_epoch + 1, "history": hist, "test": test,
            "best_val_score": best_score}, best_state
