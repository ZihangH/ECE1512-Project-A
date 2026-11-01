#!/bin/bash
# Ablations (3 seeds): pseudo-bag count M in {2,10} and no stop-gradient, per dataset.
# Usage: ./run_ablations.sh C16
d=$1
PY=../.venv/bin/python
export OMP_NUM_THREADS=4
$PY run_experiment.py --dataset $d --methods pbmil --seeds 42 43 44 --device cpu --M 10
$PY run_experiment.py --dataset $d --methods pbmil --seeds 42 43 44 --device cpu --no-detach
$PY run_experiment.py --dataset $d --methods pbmil --seeds 42 43 44 --device cpu --M 2
