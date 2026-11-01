# ECE1512 Project A: ABMIL on WSI features

Zihang Huang and Huanchi Wang, University of Toronto, ECE1512.

This repo has our code and results for gated attention-based MIL (ABMIL) on pre-extracted patch features of
CAMELYON16 (C16), CAMELYON17 (C17) and BRACS. The report is submitted separately as a PDF.

We ran the ABMIL baseline and two extension ideas:

* **Null-option attention (Proposal A, main extension):** one extra learnable "null" entry in the attention softmax
  with a zero value vector. It was run on all three datasets. The gain is small. Details are in
  [`docs/NULL_EXTENSION.md`](docs/NULL_EXTENSION.md).
* **Focal-Mass Readout (Proposal B):** a window-summed score on the patch grid. It did not beat its controls,
  so it is a negative result. We also tried pseudo-bag ABMIL (`pbmil`), which made no difference.

## Results

Test set, mean ± std over 5 training seeds (42-46). AUC and F1 are macro, all numbers x100.

| Dataset | Method | AUC | Acc | F1 |
|---|---|---|---|---|
| C16 | ABMIL | 90.8 ± 5.0 | 87.1 ± 4.1 | 85.6 ± 4.7 |
| C16 | + null option | 94.2 ± 2.2 | 91.5 ± 2.6 | 90.6 ± 3.1 |
| C17 | ABMIL | 84.0 ± 2.3 | 79.4 ± 2.1 | 56.3 ± 2.6 |
| C17 | + null option | 87.0 ± 0.7 | 81.5 ± 1.6 | 59.3 ± 4.3 |
| BRACS | ABMIL | 80.7 ± 0.8 | 39.8 ± 2.1 | 32.2 ± 3.2 |
| BRACS | + null option | 80.2 ± 0.9 | 41.4 ± 3.0 | 34.9 ± 3.4 |

The C16 difference is not significant, the C17 one is only borderline (p = 0.03, uncorrected), and BRACS shows no
AUC gain. All runs, including the controls, are in `results/summary.md` and `results/per_seed.md`.

## Files

```text
abmil.py                  baseline model
mil_lib.py                data loading, training and evaluation loop, build_model()
ext_null.py               null-option model and its controls
ext_fmr.py                Focal-Mass Readout
ext_pbmil.py              pseudo-bag ABMIL
run_experiment.py         run one method for several seeds on one dataset -> results/*.json
analyze_results.py        tables, paired tests, loss curves, bar plots
attention_eval.py         attention maps for two C16 test slides
run_*.sh                  scripts for the FMR, null, ablation and follow-up runs
null_diagnostic.py, fmr_diagnostic.py, abstain_robust.py, make_mech_tables.py   analysis scripts
PREREG_mechanism.md       follow-up study plan, written before the runs
ECE1512_ProjectA_Student.ipynb   completed notebook (C16); executed/ has the C17 and BRACS versions
results/                  per-run JSON and summary tables
report/figs/              figures
dataloader.py, utils.py, wsi_visualization.py, config/, dataset_csv/   course-provided
```

## Who did what

* Zihang Huang: baseline, data loading and training loop, analysis scripts, attention evaluation, notebook,
  Proposal B and pseudo-bag ABMIL.
* Huanchi Wang: Proposal A (null option), its controls, and the follow-up study.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on macOS (arm64), CPU only. A 30-epoch run takes about 4-8 minutes.

## Data (not in this repo)

* Features (Zenodo, https://zenodo.org/records/23130543): put the HDF5 files at
  `feature/{C16,C17,BRACS}/patch_feats_pretrain_medical_ssl.h5`, or change `data_dir` in the YAML files in `config/`.
* Slides for the attention figures (Hugging Face, `LeonYe/ECE1512-WSI-Feature`): `slide/test_067.tif`,
  `slide/test_068.tif`, `slide/test_068.xml`.

## Reproducing

Run from the repo root. Splits always use seed 42; only the training seed changes.

```bash
# baseline and pseudo-bag
for d in C16 C17 BRACS; do
  python run_experiment.py --dataset $d --methods abmil pbmil --seeds 42 43 44 45 46 --device cpu
done
# Proposal B
for d in C16 C17 BRACS; do ./run_fmr.sh $d; done
python fmr_diagnostic.py
# pseudo-bag ablations
for d in C16 C17 BRACS; do ./run_ablations.sh $d; done
# Proposal A: see docs/NULL_EXTENSION.md
# tables and figures
python analyze_results.py
python attention_eval.py --seeds 42 43 44 45 46
```

Notebook: `DATASET=C16 jupyter nbconvert --execute --to notebook --inplace ECE1512_ProjectA_Student.ipynb`
(`DATASET` can be `C16`, `C17` or `BRACS`). Its numbers (seed 42) can differ a little from the multi-seed scripts.

Training setup, the same for all methods: batch size 1, AdamW (lr 1e-4, weight decay 1e-5), 30 epochs. The epoch
with the best validation AUC + F1 is kept and the test set is evaluated once at that epoch.
See [`CODE_CHECKLIST.md`](CODE_CHECKLIST.md) for the reproducibility checklist.

Known limits: the C16 and C17 validation sets are tiny (27 and 30 slides), so model selection is noisy, and C17
runs are not bit-for-bit reproducible.
