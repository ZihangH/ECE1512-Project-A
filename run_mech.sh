#!/bin/bash
# Mechanism ablation (A) + ABMIL checkpoints for B/C. Usage: ./run_mech.sh C16
d=$1
PY=../.venv/bin/python
export OMP_NUM_THREADS=4
$PY run_experiment.py --dataset $d --methods null_nolog null_val null_k4val --seeds 42 43 44 45 46 --device cpu --save-ckpt
# ABMIL re-run into a separate dir to get checkpoints (also checks determinism vs results/*.json)
$PY run_experiment.py --dataset $d --methods abmil --seeds 42 43 44 45 46 --device cpu --save-ckpt --out results_abmil_ckpt
# extra: learnable null value with 100x lr (the default lr barely moves it, see null_val)
$PY run_experiment.py --dataset $d --methods null_valhi --seeds 42 43 44 45 46 --device cpu --save-ckpt
# k4val crashed on the first launch (logging bug for K>1, fixed) -> rerun
$PY run_experiment.py --dataset $d --methods null_k4val --seeds 42 43 44 45 46 --device cpu --save-ckpt
