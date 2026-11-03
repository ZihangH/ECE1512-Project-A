"""Extension study B (evidence as abstention score) and C (bag-size robustness).

Needs checkpoints:  results_abmil_ckpt/<D>_abmil_seed*.pt,  results/<D>_{null,null_nolog,...}_seed*.pt
(produced by run_mech.sh / run_null.sh).  Writes results/abstain_robust.json and .md.

B. For every (dataset, model, seed) take the test-set predictions and a confidence score, and ask
   (i) how well does the score detect the model's own mistakes (AUROC of -score vs "wrong"),
   (ii) risk-coverage: keep the most-confident fraction c of slides, report error rate; AURC = mean
        error over all coverage levels (lower is better); also error at 80 % and 50 % coverage.
   Scores:  MSP = max softmax prob (all models);  E = 1 - a_0 = sum_i a_i  (null models only).
   NB  E is *higher* for slides with more evidence; we treat high E as "confident" (error score = -E).
C. Test-time patch subsampling: keep a random fraction f in {1, .5, .25, .1, .05} of the patches of every test slide
   (R = 5 independent draws; the SAME index sets are used for every model), report test AUC / acc and
   the mean evidence E.  Min 8 patches are kept.
Usage: ../.venv/bin/python abstain_robust.py [C16 C17 BRACS]
"""
import glob, json, sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import mil_lib as L
from utils import classification_metrics

MODELS = {"abmil": "results_abmil_ckpt", "null": "results", "null_fixed": "results",
          "null_nolog": "results", "null_val": "results", "null_valhi": "results", "null_k4val": "results"}
FRACS = [1.0, 0.5, 0.25, 0.1, 0.05]
R = 5


def seed_of(p):
    return int(p.rsplit("seed", 1)[1][:-3])


@torch.no_grad()
def forward_all(model, ds, idx_fn=None):
    model.eval()
    probs, ys, ev = [], [], []
    for i in range(len(ds)):
        d = ds[i]
        x = d["input"].unsqueeze(0)
        if idx_fn is not None:
            x = x[:, idx_fn(i, x.shape[1])]
        if hasattr(model, "null_logit"):
            lo, _, e = model(x, return_attention=True)
            ev.append(float(e))
        else:
            lo = model(x)
            ev.append(np.nan)
        probs.append(torch.softmax(lo, 1)[0]); ys.append(int(d["label"]))
    return torch.stack(probs), torch.tensor(ys), np.array(ev)


def risk_coverage(conf, wrong):
    order = np.argsort(-conf, kind="stable")
    w = wrong[order].astype(float)
    risk = np.cumsum(w) / np.arange(1, len(w) + 1)
    n = len(w)
    return float(risk.mean()), float(risk[int(np.ceil(0.8 * n)) - 1]), float(risk[int(np.ceil(0.5 * n)) - 1])


def err_auc(conf, wrong):
    if wrong.sum() in (0, len(wrong)):
        return np.nan
    return float(roc_auc_score(wrong, -conf))


def main():
    dsets = sys.argv[1:] or ["C16", "C17", "BRACS"]
    out = {"B": {}, "C": {}}
    for d in dsets:
        conf, tr, va, te = L.load_data(d, ".")
        n_te = len(te)
        for m, folder in MODELS.items():
            paths = sorted(glob.glob(f"{folder}/{d}_{m}_seed*.pt"))
            if not paths:
                continue
            recB, recC = [], {f: [] for f in FRACS}
            for p in paths:
                s = seed_of(p)
                model = L.build_model(conf, m)
                model.load_state_dict(torch.load(p))
                # ---------- B ----------
                probs, y, ev = forward_all(model, te)
                msp = probs.max(1).values.numpy()
                wrong = (probs.argmax(1) != y).numpy()
                r = {"seed": s, "err_rate": float(wrong.mean()),
                     "msp": dict(zip(("aurc", "risk80", "risk50"), risk_coverage(msp, wrong)),
                                 err_auc=err_auc(msp, wrong))}
                if not np.isnan(ev).any():
                    r["E"] = dict(zip(("aurc", "risk80", "risk50"), risk_coverage(ev, wrong)),
                                  err_auc=err_auc(ev, wrong))
                    r["E_or_msp_rank_corr"] = float(np.corrcoef(np.argsort(np.argsort(ev)), np.argsort(np.argsort(msp)))[0, 1])
                recB.append(r)
                # ---------- C ----------
                for f in FRACS:
                    per = []
                    for k in range(R):
                        def idx_fn(i, n, f=f, k=k):
                            g = torch.Generator().manual_seed(1000003 * k + 7919 * i + 13)
                            keep = max(8, int(round(f * n))) if f < 1 else n
                            return torch.randperm(n, generator=g)[:min(keep, n)].sort().values
                        pr, yy, e = forward_all(model, te, idx_fn if f < 1 else None)
                        mt = classification_metrics(pr, yy, conf.n_class)
                        mt["evidence"] = float(np.nanmean(e)) if not np.isnan(e).all() else None
                        per.append(mt)
                        if f == 1.0:
                            break   # full bag is deterministic
                    recC[f].append({"seed": s, **{k2: (float(np.mean([q[k2] for q in per])) if per[0][k2] is not None else None)
                                                  for k2 in ("auc", "acc", "f1", "evidence")}})
                print(f"{d} {m} seed {s} done", flush=True)
            out["B"].setdefault(d, {})[m] = recB
            out["C"].setdefault(d, {})[m] = {str(f): v for f, v in recC.items()}
    json.dump(out, open("results/abstain_robust.json", "w"), indent=1)
    write_md(out)


def ms(v, scale=1.0):
    v = np.array([x for x in v if x is not None and not np.isnan(x)], float) * scale
    return f"{v.mean():.3f}±{v.std(ddof=1) if len(v) > 1 else 0:.3f}" if len(v) else "n/a"


def write_md(out):
    L_ = ["# Extension B/C: abstention and bag-size robustness\n",
          "## B. Error detection / selective prediction (test set, mean±std over seeds)\n",
          "| Dataset | Model | Score | err rate | AUROC(err) | AURC↓ | risk@80%cov | risk@50%cov |", "|---|---|---|---|---|---|---|---|"]
    for d, mm in out["B"].items():
        for m, recs in mm.items():
            er = ms([r["err_rate"] for r in recs], 1)
            for sc in ("msp", "E"):
                if sc not in recs[0]:
                    continue
                L_.append(f"| {d} | {m} | {sc.upper()} | {er} | {ms([r[sc]['err_auc'] for r in recs])} | "
                          f"{ms([r[sc]['aurc'] for r in recs])} | {ms([r[sc]['risk80'] for r in recs])} | {ms([r[sc]['risk50'] for r in recs])} |")
    L_ += ["\n## C. Test-time patch subsampling: AUC (macro) / acc % / mean evidence E\n",
           "| Dataset | Model | " + " | ".join(f"{int(f*100)}%" for f in FRACS) + " |", "|---|---|" + "---|" * len(FRACS)]
    for d, mm in out["C"].items():
        for m, byf in mm.items():
            cells = []
            for f in FRACS:
                v = byf[str(f)]
                e = ms([x["evidence"] for x in v]) if v[0]["evidence"] is not None else "–"
                cells.append(f"{ms([x['auc'] for x in v])} / {ms([x['acc'] for x in v])} / E={e}")
            L_.append(f"| {d} | {m} | " + " | ".join(cells) + " |")
    open("results/abstain_robust.md", "w").write("\n".join(L_) + "\n")


if __name__ == "__main__":
    main()
