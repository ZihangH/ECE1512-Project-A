#!/bin/bash
# Focal-Mass Readout (FMR) and its two controls, 5 seeds, one dataset.
#   fmr         : multi-scale focal-mass features from patch coordinates
#   fmr_shuffle : same computation, patch->cell assignment randomly permuted (control)
#   fmr_total   : single size-only feature log(1 + sum of scores) (control)
# Usage: ./run_fmr.sh C16      (then, once for all datasets: python fmr_diagnostic.py)
d=$1
PY=../.venv/bin/python
export OMP_NUM_THREADS=4
$PY run_experiment.py --dataset $d --methods fmr fmr_shuffle fmr_total --seeds 42 43 44 45 46 --device cpu
