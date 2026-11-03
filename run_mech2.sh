#!/bin/bash
# Second half of the mechanism ablation (valhi + k4val). Usage: ./run_mech2.sh C16
d=$1
PY=../.venv/bin/python
export OMP_NUM_THREADS=4
$PY run_experiment.py --dataset $d --methods null_valhi null_k4val --seeds 42 43 44 45 46 --device cpu --save-ckpt
