# Pre-registered hypotheses for the mechanism / abstention / robustness study
(written and committed BEFORE any of these runs were started; Oct 26 2026)

## A. Mechanism ablation (null variants; identical protocol, seeds 42-46)
Variants: `null_nolog` (e_0 = b, no log N), `null_val` (learnable null value vector, 1 null),
`null_k4val` (4 nulls, each with its own logit and learnable value vector).
Note: K zero-valued nulls are *algebraically* one null with b_eff = logsumexp(b_k), so they cannot
differ from `null` except through init/optimisation; hence K>1 is only tested with learnable values.
- H-A1: `null_nolog` is WORSE than `null` in AUC on C16 and C17 (mean over seeds), because at init
  a_0 ~ 1/N ~ 1e-3 and b (lr 1e-4) cannot move it; i.e. it behaves like plain ABMIL (AUC within ~1 pt of ABMIL).
  If instead it matches `null`, the claim "the log N null competitor is what helps" is falsified and must be revised.
- H-A2: `null_val` and `null_k4val` are NOT better than `null` by more than the seed std (no extra benefit from
  a learnable default embedding); a mild degradation is possible (extra capacity, 30 epochs, tiny val set).
## B. Evidence as an abstention score (C16 only has a binary task; all 3 datasets reported)
Score s = E = 1 - a_0 (null models) vs max-softmax probability (MSP), for predicting "this prediction is wrong".
- H-B1: E is NOT a better error detector than MSP (AUROC_err(E) <= AUROC_err(MSP)) on average; E measures
  tumour burden, not decision confidence. A tumour-vs-normal confusion is expected to be uncorrelated with E.
- H-B2: risk-coverage AURC of null+MSP is similar to ABMIL+MSP (null's accuracy edge carries over, no more).
## C. Bag-size robustness (test-time random patch subsampling to 100/50/25/10/5 %, 5 draws per slide)
- H-C1: the null model degrades LESS than ABMIL at small fractions on C16 (the log N offset makes a_0 bag-size invariant),
  but the difference is within noise on C17/BRACS.
- H-C2: `null_nolog` degrades the most (a_0 ~ 1/N grows as N shrinks -> evidence drifts with bag size).

## Amendment (Oct 26, 23:55, after seeing null_nolog/null_val results on C16/C17 but before null_valhi was run)
Observation: `null_val` is indistinguishable from `null` (AUC 94.2±2.2 / 87.0±0.7) because the null value vector
moves only to norm ~0.1 under the shared lr 1e-4 (post-hoc check on checkpoints), i.e. the test was under-powered.
Added `null_valhi` (same, but lr x100 on the null value vector only).
- H-A3: `null_valhi` does not beat `null` by more than one seed std (no benefit from a learned default embedding).
