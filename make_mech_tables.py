"""Build report/tab_mech.tex, tab_abstain.tex, tab_robust.tex from results/*.json (extension A, B, C)."""
import json, glob, numpy as np
from scipy import stats
DS = [("C16", "C16"), ("C17", "C17"), ("BRACS", "BRACS")]
SEEDS = [42, 43, 44, 45, 46]

def auc(method, d, folder="results"):
    return np.array([100 * json.load(open(f"{folder}/{d}_{method}_seed{s}.json"))["test"]["auc"] for s in SEEDS])

def fmt(x): return f"{x.mean():.1f}$\\pm${x.std(ddof=1):.1f}"
def dp(a, ref):
    d = a - ref
    if np.allclose(d, 0): return "$+0.0$ (--)"
    p = stats.ttest_rel(a, ref).pvalue
    ps = "<.001" if p < .001 else f"{p:.3f}".lstrip("0")
    return f"${d.mean():+.1f}$ ({ps})"

# ---------- A ----------
def dstar(a, ref):
    d = a - ref
    if np.allclose(d, 0): return "$\\pm0.0$"
    p = stats.ttest_rel(a, ref).pvalue
    return f"${d.mean():+.1f}$" + ("$^{*}$" if p < .05 else "")
rows = [("ABMIL", "abmil", "results"), ("ABMIL rep.", "abmil", "results_abmil_ckpt"),
        ("Null (main)", "null", "results"), ("No $\\log N$", "null_nolog", "results"),
        ("Learned val.", "null_val", "results"), ("Val., lr$\\times$100", "null_valhi", "results"),
        ("4 nulls+val.", "null_k4val", "results")]
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Mechanism ablation: test AUC$\times$100 (mean$\pm$std over five seeds) and $\Delta$, the mean paired difference to the main null model. $^{*}$: uncorrected paired $t$-test $p<.05$ (exact $p$ in \texttt{results/summary.md}); ``rep.'' = a second run of the same ABMIL configuration (see text).}",
     r"\label{tab:mech}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
     r"\begin{tabular}{lcrcrcr}", r"\toprule",
     r" & \multicolumn{2}{c}{C16} & \multicolumn{2}{c}{C17} & \multicolumn{2}{c}{BRACS} \\",
     r"Variant & AUC & $\Delta$ & AUC & $\Delta$ & AUC & $\Delta$ \\", r"\midrule"]
for name, m, folder in rows:
    cells = []
    for d, _ in DS:
        a = auc(m, d, folder); n = auc("null", d)
        cells += [fmt(a), "--" if m == "null" else dstar(a, n)]
    L.append(f"{name} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("report/tab_mech.tex", "w").write("\n".join(L) + "\n")

# ---------- B ----------
B = json.load(open("results/abstain_robust.json"))["B"]
def agg(ds, model, score, key):
    v = [r[score][key] for r in B[ds][model]]
    return np.mean(v), np.std(v, ddof=1)
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Error detection and selective prediction on the test set (mean$\pm$std over five seeds). AUROC$_\text{err}$: ability of the score to rank wrong predictions as less trustworthy than right ones (0.5 = no information). AURC: area under the risk--coverage curve (lower is better). ABMIL uses the repeat-run checkpoints.}",
     r"\label{tab:abstain}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
     r"\begin{tabular}{llcccc}", r"\toprule",
     r"Dataset & Model, score & Err. rate & AUROC$_\text{err}$ & AURC & risk@80\% \\", r"\midrule"]
first = True
for d, name in DS:
    if not first: L.append(r"\midrule")
    first = False
    for model, mname, score, sname in [("abmil", "ABMIL", "msp", "MSP"), ("null", "Null", "msp", "MSP"), ("null", "Null", "E", "$E{=}1{-}a_0$")]:
        key = score
        er = np.array([r["err_rate"] for r in B[d][model]])
        au = agg(d, model, key, "err_auc"); ar = agg(d, model, key, "aurc"); r8 = agg(d, model, key, "risk80")
        L.append(f"{name} & {mname}, {sname} & {er.mean():.3f} & {au[0]:.2f}$\\pm${au[1]:.2f} & {ar[0]:.3f}$\\pm${ar[1]:.3f} & {r8[0]:.3f}$\\pm${r8[1]:.3f} \\\\")
        name = ""
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("report/tab_abstain.tex", "w").write("\n".join(L) + "\n")

# ---------- C ----------
C = json.load(open("results/abstain_robust.json"))["C"]
FR = ["1.0", "0.5", "0.25", "0.1", "0.05"]
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Test-time patch subsampling: macro AUC$\times$100 (mean over five seeds and five random draws) when a random fraction of each slide's patches is kept, and the loss from 100\% to 5\% (points). Identical patch subsets are used for all models. ABMIL uses the repeat-run checkpoints.}",
     r"\label{tab:robust}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
     r"\begin{tabular}{llccccc|c}", r"\toprule",
     r"Dataset & Model & 100\% & 50\% & 25\% & 10\% & 5\% & $\Delta_{100\to5}$ \\", r"\midrule"]
first = True
for d, name in DS:
    if not first: L.append(r"\midrule")
    first = False
    for model, mname in [("abmil", "ABMIL"), ("null", "Null"), ("null_nolog", "No $\\log N$")]:
        v = [100 * np.mean([r["auc"] for r in C[d][model][f]]) for f in FR]
        L.append(f"{name} & {mname} & " + " & ".join(f"{x:.1f}" for x in v) + f" & ${v[4]-v[0]:+.1f}$ \\\\")
        name = ""
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("report/tab_robust.tex", "w").write("\n".join(L) + "\n")
print("ok")
