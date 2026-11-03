#!/bin/bash
# Null-option attention (main extension) and its two controls, 5 seeds, one dataset.
#   null       : a_i = exp(e_i) / (exp(b + log N) + sum_j exp(e_j)), b learnable, null value = 0
#   null_fixed : same with b frozen at 0   (control: is the *learned* null level needed?)
#   sigmoid    : a_i = sigmoid(e_i) / N     (control: un-normalised attention, no softmax)
# Usage: ./run_null.sh C16     (then: python null_diagnostic.py)
d=$1
PY=../.venv/bin/python
export OMP_NUM_THREADS=4
$PY run_experiment.py --dataset $d --methods null null_fixed sigmoid --seeds 42 43 44 45 46 --device cpu --save-ckpt
