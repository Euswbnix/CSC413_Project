"""LTC 24 vs 48 ODE sub-steps: the one-off diagnostic of docs/preregistration_2026-09-22.md 9.3.

Added after the validation results were seen, so it is not the pre-registered section-6 check
(that one, 6/12/24, is ltc_unfolds_check.py and is done). Fixed weights: seed 0's best checkpoint
of the final 24-step LTC, loaded unchanged into an eager cell with 24 and with 48 sub-steps, on
the validation target frames only. Conditions: full history, and 75% dropped with the five shared
masks (seeds 7, 1007, 2007, 3007, 4007). Rule: |48 - 24| of the mean macro MAE <= 0.1 deg in BOTH
conditions -> "stable under this check" (not proof of convergence for every weight or gap);
otherwise "integration sensitivity unresolved" -- no retraining, no 96 steps. Everything is
reported, not only the verdict. The stored (compiled) 24-step evaluation is shown for reference.

    python hdd/ltc_unfolds48_check.py --cache cache/dinov2_10hz \
        --ckpt d4/dinov2/ckpt/ltc_lr0.001_s0_u24.pt --stored d4/dinov2/ltc_s0_val.json \
        --out d4/analysis/ltc_unfolds_24v48_seed0.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import d4_run as D
import metrics

THRESHOLD = 0.1
MASK_SEEDS = [1000 * m + 7 for m in range(5)]        # 7, 1007, 2007, 3007, 4007, as in score_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stored", help="the final evaluation JSON of the same checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt-unit", type=float, default=1.0, help="as the checkpoint was trained (d4_run --dt-unit)")
    a = ap.parse_args()
    D.refuse_inside_git(a.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train = D.Cached(a.cache, "train", 30, dev)
    val = D.Cached(a.cache, "val", 30, dev)
    ytr, yva = train.labels(), val.labels()
    mu, sd = float(ytr.mean()), float(ytr.std())
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)["best_state"]
    res = dict(ckpt=os.path.basename(a.ckpt), dt_unit=a.dt_unit, frames=int(len(yva)), threshold=THRESHOLD,
               mask_seeds=MASK_SEEDS, unfolds={})
    for k in (24, 48):
        t0 = time.perf_counter()
        model = D.Arm("ltc", train.items[0]["feats"].shape[1], seed=a.seed, ode_unfolds=k,
                      dt_unit=a.dt_unit).to(dev)
        model.load_state_dict(state)
        full = metrics.macro_mae(D.predict(model, val, mu, sd, 1.0, MASK_SEEDS[0]), yva)
        masks = [metrics.macro_mae(D.predict(model, val, mu, sd, 0.25, s), yva) for s in MASK_SEEDS]
        res["unfolds"][str(k)] = dict(full=full, keep_0_25=float(np.mean(masks)), keep_0_25_masks=masks)
        print(f"ode_unfolds {k}: full {full:.4f}, keep 0.25 {np.mean(masks):.4f} "
              f"(masks {' '.join(f'{m:.4f}' for m in masks)})  [{time.perf_counter() - t0:.0f}s]", flush=True)
    u24, u48 = res["unfolds"]["24"], res["unfolds"]["48"]
    res["diff_48_minus_24"] = dict(full=u48["full"] - u24["full"],
                                   keep_0_25=u48["keep_0_25"] - u24["keep_0_25"],
                                   keep_0_25_masks=[b - c for b, c in zip(u48["keep_0_25_masks"],
                                                                          u24["keep_0_25_masks"])])
    d = res["diff_48_minus_24"]
    res["stable"] = abs(d["full"]) <= THRESHOLD and abs(d["keep_0_25"]) <= THRESHOLD
    res["verdict"] = ("stable under this check" if res["stable"]
                      else "integration sensitivity unresolved")
    if a.stored:
        st = json.load(open(a.stored))["conditions"]
        res["stored_compiled_24"] = dict(full=st["keep_1"][0]["macro_mae"],
                                         keep_0_25_masks=[r["macro_mae"] for r in st["keep_0.25"]])
        print(f"stored (compiled) 24: full {res['stored_compiled_24']['full']:.4f}, keep 0.25 masks "
              f"{' '.join(f'{m:.4f}' for m in res['stored_compiled_24']['keep_0_25_masks'])}")
    print(f"48 - 24: full {d['full']:+.4f}, keep 0.25 {d['keep_0_25']:+.4f} "
          f"(per mask {' '.join(f'{m:+.4f}' for m in d['keep_0_25_masks'])}); threshold {THRESHOLD} "
          f"-> {res['verdict']}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    os.chmod(a.out, 0o600)


if __name__ == "__main__":
    main()
