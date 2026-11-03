# Proposal A - null-option attention (`NullABMIL`) and its follow-up study

Owner: Huanchi Wang. Code: `ext_null.py`, `run_null.sh`, `null_diagnostic.py`, `run_mech.sh`, `run_mech2.sh`,
`abstain_robust.py`, `make_mech_tables.py`, `PREREG_mechanism.md`. Report: Sections "Extension" and "Follow-up Study".

**Idea.** A learnable "null" entry with logit `b + log N` and a zero value vector is added to the attention softmax,
so a slide can put attention mass on "nothing" instead of being forced to spread 100 % of it over patches. It is
evaluated under the identical protocol as the baseline (5 seeds) with two controls (null logit frozen at 0 =
`null_fixed`; unnormalised sigmoid attention = `sigmoid`) and a null-mass diagnostic. The effect is **modest and
statistically weak**. The mechanism itself is not ours (it builds on softmax-one, attention bias and learnable
attention sinks; see the report's "Relation to prior work"). To our knowledge, and per a documented but
non-exhaustive search of arXiv abstracts, it has not been applied to WSI MIL; we do not claim to be first.

A follow-up study (pre-registered in [`PREREG_mechanism.md`](PREREG_mechanism.md); report Sec. "Follow-up Study")
asks (A) which ingredient of the null option matters, (B) whether the null mass is a useful abstention / error score and
(C) how the null option copes with fewer patches per slide. Short answers: (A) the `log N` offset matters, a learned null value
or several nulls do not help; (B) `E = 1 - a_0` is *not* an error detector (AUROC for errors 0.43-0.44 on C16/C17, chance level
or worse); (C) the null model does *not* degrade more gracefully, it loses at least as much AUC as ABMIL on C16/C17.

## Statistics and diagnostics (main extension)

Paired differences null - ABMIL (uncorrected paired t-test): C16 AUC +3.5 (p = .14); C17 AUC +3.0 (p = .031, 5/5 seeds);
BRACS AUC -0.5 (p = .039). Of 18 tests of null / frozen-null vs. ABMIL, 4 have p < .05 (about 1 expected by chance),
and the learned `b` moves only to about -0.02 (null and frozen-null are indistinguishable), so the gain comes from the
fixed `log N` null competitor, not from learning a bias. Sigmoid attention is clearly worse, but that control is untuned and
produces near-uniform attention, so it does not prove that softmax competition is essential. The null mass `E = 1 - a_0`
separates tumour from normal slides on C16 (AUROC 0.947 ± 0.017; `log N` alone gives 0.644) and the attention on the annotated slide
`test_068` is not harmed (patch AUROC 0.975 vs. 0.963 for ABMIL; one slide only). The `log N` offset and the null value are ablated in the follow-up study below (the initialisation of `b` is not). Details: `results/summary.md`, `results/per_seed.md`, `results/null_diagnostic.md`.

## Follow-up study (A/B/C), test AUC x100, mean +/- std over 5 seeds

| Variant | C16 | C17 | BRACS |
|---|---|---|---|
| ABMIL | 90.8 ± 5.0 | 84.0 ± 2.3 (repeat run: 85.4 ± 1.3) | 80.7 ± 0.8 |
| null (main) | 94.2 ± 2.2 | 87.0 ± 0.7 | 80.2 ± 0.9 |
| no log N (`null_nolog`) | 90.7 ± 5.1 | 84.8 ± 1.4 | 81.0 ± 0.9 |
| learned null value (`null_val`) | 94.2 ± 2.2 | 87.0 ± 0.7 | 80.2 ± 0.9 |
| learned value, lr x100 (`null_valhi`) | 93.9 ± 2.4 | 88.0 ± 1.4 | 80.8 ± 0.5 |
| 4 nulls + values (`null_k4val`) | 95.7 ± 1.2 | 87.1 ± 1.1 | 79.4 ± 1.0 |

B (error detection, AUROC for errors; 0.5 = no information): max-softmax-prob 0.78 / 0.72 / 0.70 (C16 / C17 / BRACS, null model)
vs. evidence `E` 0.44 / 0.43 / 0.67. C (random patch subsampling 100 % -> 5 %, AUC change): ABMIL -11.2 / -2.9 / -1.9,
null -13.3 / -5.2 / -1.1. Several hypotheses were *not* supported (H-C1, H-C2) and H-A3 was an amendment written after seeing
results; see `PREREG_mechanism.md`. All p-values are uncorrected. **Reproducibility caveat:** re-running ABMIL on C17 with the same
seeds did not reproduce the first run (84.0 vs 85.4 AUC; C16 and BRACS were identical), so single-configuration differences
below about 1.4 AUC on C17 are within run-to-run noise. Details: `results/abstain_robust.md`, `results/summary.md`.

## Commands (run from the repository root; needs the baseline from the main README first)

```bash
# main extension: null option + frozen-null + sigmoid controls, 5 seeds each (saves checkpoints, *.pt, git-ignored)
for d in C16 C17 BRACS; do ./run_null.sh $d; done
python null_diagnostic.py           # -> results/null_diagnostic.{json,md}
# follow-up study: A (ablation; saves checkpoints) then B and C (need the checkpoints)
for d in C16 C17 BRACS; do ./run_mech.sh $d; done     # ./run_mech2.sh is the second half of the same script
python abstain_robust.py && python make_mech_tables.py
```

`run_mech*.sh` also trains ABMIL checkpoints into `results_abmil_ckpt/` (their JSONs are in the repo, the `.pt` files are not).
