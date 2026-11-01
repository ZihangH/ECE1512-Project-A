"""Aggregate results/*.json into tables (markdown + LaTeX) and figures.

Usage: python analyze_results.py
Outputs: results/summary.md, report/tables_generated.tex, report/figs/*.pdf|png
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DATASETS = ["C16", "C17", "BRACS"]
NAMES = {"C16": "CAMELYON16", "C17": "CAMELYON17", "BRACS": "BRACS"}
METRICS = [("auc", "AUC", 100), ("acc", "Acc (%)", 1), ("f1", "F1", 100)]
os.makedirs("report/figs", exist_ok=True)


def load():
    runs = defaultdict(list)  # (dataset, config) -> list of run dicts
    for p in sorted(glob.glob("results/*.json")) + sorted(glob.glob("results_abmil_ckpt/*.json")):
        r = json.load(open(p))
        if "dataset" not in r:
            continue
        base = os.path.basename(p)[:-5]
        cfg = base[len(r["dataset"]) + 1:].rsplit("_seed", 1)[0]
        if p.startswith("results_abmil_ckpt"):
            cfg = "abmil_rerun"   # identical code/seeds, re-run to obtain checkpoints (noise-floor estimate)
        cfg = cfg[len("pbmil_"):] if cfg.startswith("pbmil_") else cfg
        runs[(r["dataset"], cfg)].append(r)
    return runs


def ms(vals):
    v = np.array(vals, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def main():
    runs = load()
    lines, tex = [], []
    ORDER = ["abmil", "abmil_rerun", "null", "null_fixed", "sigmoid", "null_nolog", "null_val", "null_valhi", "null_k4val", "fmr", "fmr_shuffle", "fmr_total", "M5", "M2", "M10", "M5_nodetach"]
    cfgs = sorted({c for _, c in runs}, key=lambda c: ORDER.index(c) if c in ORDER else 99)
    # ---- main table -------------------------------------------------------
    lines.append("| Dataset | Method | #seeds | AUC (x100) | Acc (%) | F1 (x100) | best epoch |")
    lines.append("|---|---|---|---|---|---|---|")
    for d in DATASETS:
        for c in cfgs:
            rs = runs.get((d, c))
            if not rs:
                continue
            row = [NAMES[d], c, str(len(rs))]
            for k, _, scale in METRICS:
                m, s = ms([r["test"][k] * scale for r in rs])
                row.append(f"{m:.1f} ± {s:.1f}")
            m, s = ms([r["best_epoch"] for r in rs])
            row.append(f"{m:.1f}")
            lines.append("| " + " | ".join(row) + " |")
    # ---- paired comparison -----------------------------------------------
    PAIRS = [("abmil", c) for c in cfgs if c != "abmil"] + [("abmil", "abmil_rerun"), ("abmil_rerun", "null"), ("null_fixed", "null"), ("sigmoid", "null"), ("null", "null_nolog"), ("null", "null_val"), ("null", "null_valhi"), ("null", "null_k4val"), ("fmr_shuffle", "fmr"), ("fmr_total", "fmr")]
    lines += ["", "Paired comparison (same seed = same init & data order; same split). diff = method - reference:", ""]
    lines.append("| Dataset | Method | Reference | metric | mean diff | wins/seeds | paired t p-value |")
    lines.append("|---|---|---|---|---|---|---|")
    for d in DATASETS:
        for ref, c in PAIRS:
            if (d, c) not in runs or (d, ref) not in runs:
                continue
            base = {r["seed"]: r for r in runs[(d, ref)]}
            ext = {r["seed"]: r for r in runs[(d, c)]}
            seeds = sorted(set(base) & set(ext))
            if len(seeds) < 2:
                continue
            for k, nm, scale in METRICS:
                diff = np.array([(ext[s]["test"][k] - base[s]["test"][k]) * scale for s in seeds])
                p = stats.ttest_1samp(diff, 0).pvalue if diff.std() > 0 else float("nan")
                lines.append(f"| {NAMES[d]} | {c} | {ref} | {nm} | {diff.mean():+.2f} | "
                             f"{(diff > 0).sum()}/{len(seeds)} | {p:.3f} |")
    # paired table of the null option vs ABMIL / frozen-b control / sigmoid control (for the report)
    def pcell(d, c, ref, k, scale):
        a = {r["seed"]: r for r in runs[(d, c)]}; b = {r["seed"]: r for r in runs[(d, ref)]}
        sd = sorted(set(a) & set(b))
        diff = np.array([(a[x]["test"][k] - b[x]["test"][k]) * scale for x in sd])
        p = stats.ttest_1samp(diff, 0).pvalue if diff.std() > 0 else float("nan")
        ps = "--" if np.isnan(p) else ("<.001" if p < .001 else f"{p:.3f}".lstrip("0"))
        return f"${diff.mean():+.1f}$ ({ps})"
    NP = ["\\begin{table}[t]", "\\centering",
          "\\caption{Paired differences on the test set, mean over five seeds (uncorrected paired $t$-test $p$ in brackets; ``--'' means identical accuracy and F1 on every seed, so the test is undefined). Positive means the first model is better.}",
          "\\label{tab:nullpaired}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
          "\\begin{tabular}{llccc}", "\\toprule", "Dataset & Comparison & $\\Delta$AUC & $\\Delta$Acc & $\\Delta$F1 \\\\", "\\midrule"]
    for i, d in enumerate(DATASETS):
        if i:
            NP.append("\\midrule")
        for j, (c, ref, nm) in enumerate([("null", "abmil", "Null $-$ ABMIL"), ("null", "null_fixed", "Null $-$ frozen $b$"),
                                          ("null", "sigmoid", "Null $-$ sigmoid")]):
            first = f"\\multirow{{3}}{{*}}{{{ {'C16': 'C16', 'C17': 'C17', 'BRACS': 'BRACS'}.get(d, d) }}}" if j == 0 else ""
            cells = [pcell(d, c, ref, k, sc) for k, _, sc in METRICS]
            NP.append(f"{first} & {nm} & " + " & ".join(cells) + " \\\\")
    NP += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open("report/tab_nullpaired.tex", "w").write("\n".join(NP) + "\n")
    open("results/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- LaTeX tables for the report ---------------------------------------
    RUNS = runs
    LABEL = {"abmil": "ABMIL", "null": "ABMIL + null option", "null_fixed": "Null, $b{=}0$ frozen",
             "sigmoid": "Sigmoid attention", "fmr": "ABMIL + FMR", "fmr_shuffle": "FMR, shuffled coords",
             "fmr_total": "ABMIL + total mass", "M5": "PB-ABMIL ($M{=}5$)", "M2": "PB-ABMIL ($M{=}2$)",
             "M10": "PB-ABMIL ($M{=}10$)", "M5_nodetach": "PB-ABMIL ($M{=}5$, no detach)"}

    def table(cfg_list, name, caption, label, seeds=None):
        runs = {k: [r for r in v if seeds is None or r["seed"] in seeds] for k, v in RUNS.items()}
        out = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
               "\\begin{tabular}{llcccc}", "\\toprule",
               "Dataset & Method & AUC & Acc (\\%) & F1 & Best ep. \\\\", "\\midrule"]
        for di, d in enumerate(DATASETS):
            avail = [c for c in cfg_list if runs.get((d, c))]
            best = {}
            for k, _, scale in METRICS:
                best[k] = max(ms([r["test"][k] * scale for r in runs[(d, c)]])[0] for c in avail) if avail else None
            for ci, c in enumerate(avail):
                rs = runs[(d, c)]
                cells = []
                for k, _, scale in METRICS:
                    m, sd = ms([r["test"][k] * scale for r in rs])
                    txt = f"{m:.1f}$\\pm${sd:.1f}"
                    cells.append(f"\\textbf{{{txt}}}" if len(avail) > 1 and abs(m - best[k]) < 1e-9 else txt)
                cells.append(f"{ms([r['best_epoch'] for r in rs])[0]:.1f}")
                dn = f"\\multirow{{{len(avail)}}}{{*}}{{{NAMES[d]}}}" if ci == 0 else ""
                out.append(f"{dn} & {LABEL.get(c, c)} & " + " & ".join(cells) + " \\\\")
            if di < len(DATASETS) - 1:
                out.append("\\midrule")
        out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
        return "\n".join(out)

    texs = [table(["abmil"], "baseline",
                  "ABMIL baseline, test set (mean $\\pm$ std over training seeds; AUC and F1 are macro-averaged and $\\times100$).",
                  "tab:baseline"),
            table(["abmil", "null", "null_fixed", "sigmoid"], "ext",
                  "Baseline vs.\\ null-option ABMIL and its two controls on the same splits and seeds (mean $\\pm$ std over 5 seeds, best per dataset in bold).",
                  "tab:ext"),
            table(["abmil", "fmr", "fmr_shuffle", "fmr_total"], "fmr",
                  "Appendix: baseline vs.\\ Focal-Mass Readout (FMR) and its two controls (mean $\\pm$ std over 5 seeds, best per dataset in bold).",
                  "tab:fmrext"),
            table(["abmil", "M5"], "pb",
                  "Appendix: baseline vs.\\ pseudo-bag ABMIL (mean $\\pm$ std over 5 seeds, best per dataset in bold).",
                  "tab:pb"),
            table([c for c in ["abmil", "M2", "M5", "M10", "M5_nodetach"] if any((d, c) in runs for d in DATASETS)], "abl",
                  "Ablation over the number of pseudo-bags $M$ and the stop-gradient on tier-2 inputs, restricted to the three seeds (42--44) shared by all rows (mean $\\pm$ std, best per dataset in bold).", "tab:abl", seeds={42, 43, 44})]
    for nm, t in zip(["baseline", "ext", "fmr", "pb", "abl"], texs):
        open(f"report/tab_{nm}.tex", "w").write(t + "\n")

    # ---- per-seed table -----------------------------------------------------
    per = ["| Dataset | Method | seed | best ep | val AUC | val F1 | test AUC | test Acc | test F1 | test loss |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for d in DATASETS:
        for c in cfgs:
            for r in sorted(runs.get((d, c), []), key=lambda r: r["seed"]):
                h = r["history"][r["best_epoch"] - 1]
                t = r["test"]
                per.append(f"| {NAMES[d]} | {c} | {r['seed']} | {r['best_epoch']} | {h['val_auc']:.3f} | "
                           f"{h['val_f1']:.3f} | {t['auc']:.3f} | {t['acc']:.1f} | {t['f1']:.3f} | {t['loss']:.3f} |")
    open("results/per_seed.md", "w").write("\n".join(per) + "\n")

    # ---- figures ------------------------------------------------------------
    for method, fname in [("abmil", "loss_curves_baseline"), ("M5", "loss_curves_pbmil")]:
        fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.8))
        for ax, d in zip(axs, DATASETS):
            rs = runs.get((d, method), [])
            if not rs:
                continue
            tr = np.array([[h["train_loss"] for h in r["history"]] for r in rs])
            va = np.array([[h["val_loss"] for h in r["history"]] for r in rs])
            ep = np.arange(1, tr.shape[1] + 1)
            for arr, lab, col in [(tr, "train", "tab:blue"), (va, "validation", "tab:orange")]:
                ax.plot(ep, arr[0], color=col, lw=1.8, label=f"{lab} (seed 42)")
                if len(rs) > 1:
                    ax.fill_between(ep, arr.min(0), arr.max(0), color=col, alpha=0.15)
            ax.axvline(rs[0]["best_epoch"], color="gray", ls=":", lw=1)
            ax.set_title(NAMES[d], fontsize=9)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.2)
            if method == "M5":
                ax.set_yscale("log")
        axs[0].set_ylabel("Loss" + (" (train = tier1 + tier2)" if method == "M5" else ""))
        axs[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"report/figs/{fname}.pdf")
        fig.savefig(f"report/figs/{fname}.png", dpi=150)
        plt.close(fig)

    # bar charts (per-seed dots): null option + controls, and (appendix) FMR + controls
    def bars(BARS, fname):
        fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.8))
        n = len(BARS)
        for ax, (k, nm, scale) in zip(axs, METRICS):
            w = 0.8 / n
            for i, d in enumerate(DATASETS):
                for j, (c, col, lab) in enumerate(BARS):
                    rs = runs.get((d, c), [])
                    if not rs:
                        continue
                    v = np.array([r["test"][k] * scale for r in rs])
                    x = i + (j - (n - 1) / 2) * w
                    ax.bar(x, v.mean(), w, yerr=v.std(ddof=1) if len(v) > 1 else 0,
                           color=col, alpha=0.7, capsize=2, label=lab if i == 0 else None)
                    ax.scatter(np.full(len(v), x), v, s=6, color="k", zorder=3)
            ax.set_xticks(range(3))
            ax.set_xticklabels([NAMES[d] for d in DATASETS], fontsize=8)
            ax.set_title("Test " + nm, fontsize=9)
            ax.grid(axis="y", alpha=0.2)
        axs[0].legend(fontsize=6, loc="lower left")
        fig.tight_layout()
        fig.savefig(f"report/figs/{fname}.pdf")
        fig.savefig(f"report/figs/{fname}.png", dpi=150)
        plt.close(fig)

    bars([("abmil", "tab:gray", "ABMIL"), ("null", "tab:red", "+null option"),
          ("null_fixed", "tab:orange", "null, $b$=0 frozen"), ("sigmoid", "tab:blue", "sigmoid attention")],
         "test_metrics_bar")
    bars([("abmil", "tab:gray", "ABMIL"), ("fmr", "tab:red", "+FMR"),
          ("fmr_shuffle", "tab:orange", "FMR, shuffled coords"), ("fmr_total", "tab:blue", "+total mass")],
         "fmr_metrics_bar")


if __name__ == "__main__":
    main()
