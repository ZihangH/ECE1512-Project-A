"""Coordinate-permutation diagnostic for the Focal-Mass Readout (FMR) models.

For every saved checkpoint of `fmr` and `fmr_shuffle` we evaluate the TEST set twice:
  (a) with the true patch coordinates,
  (b) with the patch->grid-cell assignment randomly permuted (5 permutation seeds, averaged).
If a model really uses spatial contiguity, (b) should hurt it; a spatially blind model is unchanged.
We also report the fraction of slides whose predicted class flips, and the mean
absolute change of the predicted probability vector, plus the FMR features themselves.

Usage: python fmr_diagnostic.py     (needs results/*_fmr*_seed*.pt from run_experiment.py --save-ckpt)
Writes results/fmr_diagnostic.json and results/fmr_diagnostic.md
"""
import glob, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mil_lib as L
from ext_fmr import FocalMassABMIL
from utils import classification_metrics

DATASETS = ["C16", "C17", "BRACS"]
N_PERM = 5


@torch.no_grad()
def predict(model, ds, perm_seed=None, true_coords=False):
    model.eval()
    probs, labels, feats = [], [], []
    saved = (model.variant, model.permute_coords_at_test, model.seed_eval)
    if true_coords:
        model.variant, model.permute_coords_at_test = "fmr", False
    elif perm_seed is not None:
        model.variant, model.permute_coords_at_test, model.seed_eval = "fmr", True, perm_seed
    for i in range(len(ds)):
        d = ds[i]
        logits, f, _ = model(d["input"].unsqueeze(0), d["coords"], return_extra=True)
        probs.append(F.softmax(logits, 1).cpu()); labels.append(torch.as_tensor(d["label"]).view(1)); feats.append(f.cpu())
    model.variant, model.permute_coords_at_test, model.seed_eval = saved
    return torch.cat(probs), torch.cat(labels).long(), torch.cat(feats)


def main():
    out = {}
    for d in DATASETS:
        conf, tr, va, te = L.load_data(d, ".")
        for method, variant in [("fmr", "fmr"), ("fmr_shuffle", "shuffle")]:
            for p in sorted(glob.glob(f"results/{d}_{method}_seed*.pt")):
                seed = int(p.rsplit("seed", 1)[1][:-3])
                model = FocalMassABMIL(conf, variant)
                model.load_state_dict(torch.load(p))
                pt, y, ft = predict(model, te, true_coords=True)
                mt = classification_metrics(pt, y, conf.n_class)
                rows = []
                for ps in range(N_PERM):
                    pp, _, fp = predict(model, te, perm_seed=100 + ps)
                    m = classification_metrics(pp, y, conf.n_class)
                    rows.append({"auc": m["auc"], "acc": m["acc"], "f1": m["f1"],
                                 "flip": float((pp.argmax(1) != pt.argmax(1)).float().mean()),
                                 "dprob": float((pp - pt).abs().sum(1).mean())})
                rec = {"true": {k: mt[k] for k in ("auc", "acc", "f1")},
                       "perm": {k: float(np.mean([r[k] for r in rows])) for k in rows[0]},
                       "feat_mean_true": ft.mean(0).tolist(),
                       "feat_by_class": {int(c): ft[y == c].mean(0).tolist() for c in y.unique()}}
                out[f"{d}_{method}_seed{seed}"] = rec
                print(d, method, seed, f"AUC true {mt['auc']:.3f} -> perm {rec['perm']['auc']:.3f}; "
                      f"flip {rec['perm']['flip']:.3f}", flush=True)
    json.dump(out, open("results/fmr_diagnostic.json", "w"), indent=1)

    lines = ["| Dataset | Model (test-time) | AUC true | AUC permuted | Δ AUC | Acc true | Acc permuted | pred. flips | mean |Δp| |",
             "|---|---|---|---|---|---|---|---|---|"]
    for d in DATASETS:
        for method in ["fmr", "fmr_shuffle"]:
            rs = [v for k, v in out.items() if k.startswith(f"{d}_{method}_seed")]
            if not rs: continue
            f = lambda a, b, s=100: np.mean([r[a][b] for r in rs]) * s
            lines.append(f"| {d} | {method} | {f('true','auc'):.1f} | {f('perm','auc'):.1f} | "
                         f"{f('perm','auc')-f('true','auc'):+.1f} | {f('true','acc',1):.1f} | {f('perm','acc',1):.1f} | "
                         f"{f('perm','flip'):.3f} | {f('perm','dprob',1):.3f} |")
    open("results/fmr_diagnostic.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
