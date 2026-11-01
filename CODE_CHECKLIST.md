# Code Reproducibility Checklist (ECE1512 Project A, Zihang Huang and Huanchi Wang)

The project has **two separate artefacts**: the PDF report and this code repository.
Each has its own checklist. This file covers the **code**; the report's checklist is
in the appendix of the separately submitted PDF report.
Format follows the ML Code Completeness / Reproducibility Checklist (Pineau et al., v2.0).

| # | Item | Answer | Comment / where |
|---|------|--------|-----------------|
| 1 | Dependencies specified (libraries, versions) | Yes | `requirements.txt` pins the exact versions; Python 3.12. |
| 2 | Training code provided | Yes | `mil_lib.py` (`train_one_epoch`, `run_one`), `run_experiment.py`, and the notebook `ECE1512_ProjectA_Student.ipynb`. |
| 3 | Evaluation code provided | Yes | `mil_lib.evaluate` and `utils.classification_metrics` (accuracy, macro AUC, macro F1); `analyze_results.py` aggregates results. |
| 4 | (Pre-)trained models provided | Partly | Checkpoints are **not** committed (size and git hygiene); `run_experiment.py --save-ckpt` / `attention_eval.py` regenerate them in minutes on CPU. The frozen patch features come from the course (Zenodo/HF), not from us. |
| 5 | README with table of results and exact commands | Yes | `README.md`. |
| 6 | Data download instructions | Yes | `README.md` (Zenodo / Hugging Face links from the course handout). The data is not redistributed here. |
| 7 | Random seeds fixed | Yes | Data split seed 42 (fixed); training seeds 42-46; loader shuffling uses a seeded generator. |
| 8 | Hyper-parameters in config files / CLI | Yes | `config/*.yml`; CLI flags `--M`, `--no-detach`, `--epochs`, `--seeds`. |
| 9 | Model selection protocol clear | Yes | Best epoch by validation AUC+F1; test set evaluated once on that epoch (`mil_lib.run_one`). |
| 10 | Hardware / runtime documented | Yes | Apple M4 Pro, CPU only (`--device cpu`), about 4-6 min (ABMIL) and 6-8 min (pseudo-bag; FMR about 5-6 min on C16; null option not timed separately, expected similar to ABMIL) per 30-epoch run, measured with three runs sharing the CPU (4 threads each). See README. |
| 11 | Known non-determinism noted | Yes | CPU runs were bit-identical for a re-run with the same seed on CAMELYON16 and BRACS, but NOT on CAMELYON17 (ABMIL repeat 85.4 vs 84.0 AUC; cause unconfirmed, probably threading/float non-determinism amplified by best-epoch selection on a 30-slide validation set), so expect about 1.4 AUC of run-to-run noise on C17; results may differ slightly across platforms and torch versions. Notebook results (seed 42 only) can differ slightly from the scripted seed-42 run because of different shuffling RNG. |
| 12 | Extension code and ablations reproducible | Yes | Main extension: `NullABMIL` (modes `null`, `null_fixed`, `sigmoid`) in `ext_null.py`, `run_null.sh` and `null_diagnostic.py`. Follow-up study: modes `null_nolog`, `null_val`, `null_valhi`, `null_k4val`, `run_mech.sh`/`run_mech2.sh`, `abstain_robust.py` (error detection, patch subsampling), `make_mech_tables.py`, `PREREG_mechanism.md`. Exploratory: `FocalMassABMIL` (`ext_fmr.py`; modes `fmr`, `fmr_shuffle`, `fmr_total`), `PseudoBagABMIL` (`ext_pbmil.py`), `run_fmr.sh`, `fmr_diagnostic.py`, `run_ablations.sh`; commands in `README.md` and `docs/NULL_EXTENSION.md`. The baseline model is `abmil.py`; `mil_lib.build_model` selects the model by method name. |
