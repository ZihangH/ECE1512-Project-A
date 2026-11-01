"""Explanation check for the extension (C16 only).

Trains ABMIL and the pseudo-bag model on CAMELYON16 for several seeds, then
  1. renders the attention overlays for test_067 / test_068 of both models
     (report/figs/attn_{method}_{slide}.png, seed given by --fig-seed), and
  2. quantifies how well attention agrees with the pathologist's tumour contours
     of test_068 (the only test slide whose XML is provided with the course data):
       - patch-level AUROC of attention vs "patch centre inside tumour contour"
       - precision of the top-5 % attention patches
       - fraction of total attention mass that falls inside the contours.
Results -> results/attn_eval.json and results/attn_eval.md

Usage (from Project_A_Starter_Code):  ../.venv/bin/python attention_eval.py --seeds 42 43 44 45 46
"""
import argparse, json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py, torch, torch.nn as nn
from matplotlib.path import Path as MPath
from sklearn.metrics import roc_auc_score

import mil_lib as ml
import wsi_visualization as wv

SLIDES = ("test_068", "test_067")
PATCH = 512  # level-0 patch footprint used by the course visualisation code


class PBWrap(nn.Module):
    """Adapter so visualize_wsi_attention() can be used with the pseudo-bag model."""

    def __init__(self, pb):
        super().__init__()
        self.pb = pb

    def forward(self, x, return_attention=False):
        logits, att = self.pb.explain(x)
        return (logits, att) if return_attention else logits

    def load_state_dict(self, sd, *a, **k):
        return self.pb.load_state_dict(sd, *a, **k)


def attention_for(model, method, feats):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats).unsqueeze(0)
        if method == "abmil":
            _, att = model(x, return_attention=True)
        else:
            _, att = model.explain(x)
    return att.reshape(-1).numpy()


def tumour_mask(coords, xml_path):
    polys = [xy for xy, g in wv.read_annotation_contours(Path(xml_path))
             if g.lower() not in {"_2", "exclusion", "exclude", "normal"}]
    ctr = coords + PATCH / 2
    inside = np.zeros(len(coords), bool)
    for xy in polys:
        inside |= MPath(xy).contains_points(ctr)
    return inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--fig-seed", type=int, default=42)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True); os.makedirs("report/figs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    conf, tr, va, te = ml.load_data("C16", ".")
    feats, coords = {}, {}
    with h5py.File(conf.data_dir, "r") as h5:
        for s in SLIDES:
            feats[s] = np.asarray(h5[s]["feat"][:], dtype=np.float32)
            coords[s] = np.asarray(h5[s]["coords"][:], dtype=np.float64)
    mask = tumour_mask(coords["test_068"], os.path.join("slide", "test_068.xml"))
    print(f"test_068: {mask.sum()} / {len(mask)} patches inside tumour contours")

    out = {"n_patches": int(len(mask)), "n_tumour_patches": int(mask.sum()), "runs": []}
    for method in ("abmil", "pbmil"):
        for seed in args.seeds:
            ck = f"checkpoints/C16_{method}_seed{seed}.pt"
            res, state = ml.run_one(conf, tr, va, te, method=method, seed=seed,
                                    device=args.device, M=args.M, log=lambda *_: None)
            torch.save(state, ck)
            model = ml.build_model(conf, method, args.M)
            model.load_state_dict(state)
            att = attention_for(model, method, feats["test_068"])
            k = max(1, int(0.05 * len(att)))
            top = np.argsort(-att)[:k]
            rec = dict(method=method, seed=seed, best_epoch=res["best_epoch"],
                       test=res["test"],
                       att_auc=float(roc_auc_score(mask, att)),
                       top5_precision=float(mask[top].mean()),
                       mass_in_tumour=float(att[mask].sum() / att.sum()),
                       chance=float(mask.mean()))
            out["runs"].append(rec)
            print(method, seed, {k_: round(v, 4) for k_, v in rec.items() if isinstance(v, float)})
            if seed == args.fig_seed:
                wrapped = model if method == "abmil" else PBWrap(model)
                counter = iter(SLIDES)

                def save_show(method=method, counter=counter):
                    sid = next(counter)
                    plt.savefig(f"report/figs/attn_{method}_{sid}.png", dpi=110)
                    plt.close("all")
                plt.show = save_show
                wv.plt.show = save_show
                wv.visualize_wsi_attention(wrapped, conf, ".", state, slide_ids=SLIDES)
    json.dump(out, open("results/attn_eval.json", "w"), indent=1)

    lines = ["| Method | Attention AUROC | Top-5% precision | Attention mass in tumour |",
             "|---|---|---|---|"]
    for method in ("abmil", "pbmil"):
        r = [x for x in out["runs"] if x["method"] == method]
        f = lambda key: f"{np.mean([x[key] for x in r]):.3f} ± {np.std([x[key] for x in r], ddof=1) if len(r) > 1 else 0:.3f}"
        lines.append(f"| {method} | {f('att_auc')} | {f('top5_precision')} | {f('mass_in_tumour')} |")
    lines.append(f"\nChance level (fraction of tumour patches) = {out['n_tumour_patches']/out['n_patches']:.3f}")
    open("results/attn_eval.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
