"""Run ABMIL baseline / pseudo-bag extension on one dataset for several seeds.

Example:
    python run_experiment.py --dataset BRACS --methods abmil pbmil --seeds 42 43 44 45 46
Results are written to results/<dataset>_<method>[_tag]_seed<seed>.json
"""
import argparse
import json
import os
import time

import torch

import mil_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(L.CONFIGS))
    ap.add_argument("--methods", nargs="+", default=["abmil"], choices=["abmil", "pbmil", "fmr", "fmr_shuffle", "fmr_total", "null", "null_fixed", "sigmoid", "null_nolog", "null_val", "null_k4val", "null_valhi"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--M", type=int, default=5, help="number of pseudo-bags (pbmil)")
    ap.add_argument("--no-detach", action="store_true", help="ablation: no tier-2 detach")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results")
    ap.add_argument("--save-ckpt", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    conf, tr, va, te = L.load_data(a.dataset, ".")
    print(f"{a.dataset}: train {len(tr)} val {len(va)} test {len(te)} "
          f"D={conf.D_feat} C={conf.n_class} (loaded in {time.time()-t0:.0f}s)", flush=True)
    for method in a.methods:
        for seed in a.seeds:
            tag = a.tag or (f"M{a.M}" + ("_nodetach" if a.no_detach else "") if method == "pbmil" else "")
            name = f"{a.dataset}_{method}{'_' + tag if tag else ''}_seed{seed}"
            path = os.path.join(a.out, name + ".json")
            if os.path.exists(path):
                print("skip (exists)", path, flush=True)
                continue
            res, state = L.run_one(conf, tr, va, te, method, seed, a.epochs, a.device,
                                   a.M, not a.no_detach, log=lambda s: print(s, flush=True))
            res.update(dataset=a.dataset, n_class=conf.n_class,
                       split={"train": len(tr), "val": len(va), "test": len(te)})
            json.dump(res, open(path, "w"), indent=1)
            if a.save_ckpt:
                torch.save(state, os.path.join(a.out, name + ".pt"))


if __name__ == "__main__":
    main()
