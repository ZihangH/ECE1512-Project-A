"""Null-mass diagnostic for the null-option attention models (needs results/*_null*_seed*.pt).

Part 1  (all datasets, test slides) -- does the model use its "abstain" option sensibly?
    * learned null level b, and null mass a_0 = 1 - sum_i a_i per test slide
    * mean a_0 per class (C16: normal vs tumour; C17 / BRACS: per class)
    * C16: AUROC of the evidence score (1 - a_0) for tumour vs normal, compared with a
      size-only score log N (to show the evidence is not just bag size)
    * Spearman correlation of a_0 with log N (bag-size confound)
    The same quantities are computed for the `sigmoid` control (evidence = mean sigmoid score)
    and for `null_fixed` (b = 0).
Part 2  (C16, test_068 only, the one slide with pathologist contours) -- attention vs annotation
    for ABMIL (checkpoints/C16_abmil_seed*.pt), null and sigmoid models: patch AUROC,
    top-5 % precision, fraction of attention that lies inside the tumour contours.
Writes results/null_diagnostic.json and results/null_diagnostic.md
Usage: ../.venv/bin/python null_diagnostic.py
"""
import glob, json, os
import h5py
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import mil_lib as L
from attention_eval import tumour_mask

DATASETS = ["C16", "C17", "BRACS"]
METHODS = ["null", "null_fixed", "sigmoid"]


@torch.no_grad()
def slide_stats(model, ds):
    model.eval()
    ev, n, y = [], [], []
    for i in range(len(ds)):
        d = ds[i]
        x = d["input"].unsqueeze(0)
        _, a, e = model(x, return_attention=True)
        ev.append(float(e)); n.append(x.shape[1]); y.append(int(d["label"]))
    return np.array(ev), np.array(n), np.array(y)


def seed_of(p):
    return int(p.rsplit("seed", 1)[1][:-3])


def main():
    out = {"part1": {}, "part2": []}
    for d in DATASETS:
        conf, tr, va, te = L.load_data(d, ".")
        for m in METHODS:
            for p in sorted(glob.glob(f"results/{d}_{m}_seed*.pt")):
                s = seed_of(p)
                model = L.build_model(conf, m)
                model.load_state_dict(torch.load(p))
                ev, n, y = slide_stats(model, te)
                rec = {"b": float(model.null_logit.item()),
                       "evidence_mean": float(ev.mean()), "evidence_min": float(ev.min()),
                       "evidence_max": float(ev.max()),
                       "evidence_by_class": {int(c): float(ev[y == c].mean()) for c in np.unique(y)},
                       "spearman_evidence_logN": float(spearmanr(ev, np.log(n))[0])}
                if conf.n_class == 2:
                    rec["auc_evidence"] = float(roc_auc_score(y, ev))
                    rec["auc_logN"] = float(roc_auc_score(y, np.log(n)))
                out["part1"][f"{d}_{m}_seed{s}"] = rec
                print(d, m, s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()
                                if k != "evidence_by_class"}, flush=True)

    # ---- Part 2: attention vs annotation on test_068 -------------------------------------
    conf = L.load_data("C16", ".")[0]
    with h5py.File(conf.data_dir, "r") as h5:
        feats = np.asarray(h5["test_068"]["feat"][:], dtype=np.float32)
        coords = np.asarray(h5["test_068"]["coords"][:], dtype=np.float64)
    mask = tumour_mask(coords, os.path.join("slide", "test_068.xml"))
    x = torch.from_numpy(feats).unsqueeze(0)
    sources = [("abmil", f"checkpoints/C16_abmil_seed{s}.pt") for s in range(42, 47)]
    for m in METHODS:
        sources += [(m, p) for p in sorted(glob.glob(f"results/C16_{m}_seed*.pt"))]
    for m, p in sources:
        model = L.build_model(conf, m)
        model.load_state_dict(torch.load(p))
        model.eval()
        with torch.no_grad():
            o = model(x, return_attention=True)
        if len(o) == 2:                      # plain ABMIL: softmax attention, evidence = 1
            a, e = o[1], torch.tensor(1.0)
        else:
            _, a, e = o
        att = a.reshape(-1).numpy()
        e = e.reshape(-1)[0]
        top = np.argsort(-att)[:max(1, int(0.05 * len(att)))]
        out["part2"].append({"method": m, "seed": seed_of(p), "evidence_068": float(e),
                             "att_auc": float(roc_auc_score(mask, att)),
                             "top5_precision": float(mask[top].mean()),
                             "mass_in_tumour_renorm": float(att[mask].sum() / att.sum()),
                             "mass_in_tumour_abs": float(att[mask].sum()),
                             "chance": float(mask.mean())})
    json.dump(out, open("results/null_diagnostic.json", "w"), indent=1)

    # ---- markdown ----------------------------------------------------------------------
    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.3f}"
    lines = ["### Part 1: evidence (1 - null mass) on test slides", "",
             "| Dataset | Model | learned b | mean evidence | evidence range (min–max, mean over seeds) | evidence by class | ρ(evidence, log N) | AUROC evidence | AUROC log N |",
             "|---|---|---|---|---|---|---|---|---|"]
    for d in DATASETS:
        for m in METHODS:
            r = [v for k, v in out["part1"].items() if k.startswith(f"{d}_{m}_seed")]
            if not r:
                continue
            cls = sorted(r[0]["evidence_by_class"])
            by = ", ".join(f"c{c}={np.mean([x['evidence_by_class'][c] for x in r]):.2f}" for c in cls)
            auc = ms([x["auc_evidence"] for x in r]) if "auc_evidence" in r[0] else "–"
            aucn = ms([x["auc_logN"] for x in r]) if "auc_logN" in r[0] else "–"
            lines.append(f"| {d} | {m} | {np.mean([x['b'] for x in r]):.2f} | {ms([x['evidence_mean'] for x in r])} | "
                         f"{np.mean([x['evidence_min'] for x in r]):.2f}–{np.mean([x['evidence_max'] for x in r]):.2f} | {by} | "
                         f"{ms([x['spearman_evidence_logN'] for x in r])} | {auc} | {aucn} |")
    lines += ["", "### Part 2: attention vs pathologist contours on C16 test_068", "",
              "| Model | Attention AUROC | Top-5% precision | Attention mass in tumour (renormalised) | Evidence on this slide |",
              "|---|---|---|---|---|"]
    for m in ["abmil"] + METHODS:
        r = [x for x in out["part2"] if x["method"] == m]
        if r:
            lines.append(f"| {m} | {ms([x['att_auc'] for x in r])} | {ms([x['top5_precision'] for x in r])} | "
                         f"{ms([x['mass_in_tumour_renorm'] for x in r])} | {ms([x['evidence_068'] for x in r])} |")
    lines.append(f"\nChance (tumour fraction) = {out['part2'][0]['chance']:.3f}")
    open("results/null_diagnostic.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
