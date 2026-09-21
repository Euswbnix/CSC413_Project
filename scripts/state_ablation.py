#!/usr/bin/env python3
"""How much does the recurrent state actually contribute?

The per-position MAE curve says the CfC gets no benefit from 15 frames of history while a
zero-parameter causal mean gets 16.5% on the same frames. That is inferred from a curve.
This measures it directly: run the identical trained weights twice over the same split,
once carrying the state and once with the state severed before EVERY frame, and compare.

Severing is done by driving the rollout one frame at a time and passing `hx=None` on each
call, rather than by patching a cell. That keeps it arm-agnostic -- it works for the LSTM
and GRU, whose recurrence lives inside cuDNN where a forward patch cannot reach -- and it
severs *within* a chunk, which resetting between chunks does not.

If severed == carried, the state is inert and the "recurrent" arm is a per-frame CNN.
"""
import argparse, json, pathlib, sys
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import metrics
from data.dataset import SteeringData, rollout_chunks
from models.interface import build_arm


@torch.no_grad()
def rollout(model, data, split, fixed_dt, sever, chunk):
    model.eval()
    P, Y, V, hx, seg = [], [], [], None, None
    for (seg_i, s, e), b in rollout_chunks(data, split, chunk=chunk):
        if seg_i != seg:
            hx, seg = None, seg_i
        dt = torch.ones_like(b.dt) if fixed_dt else b.dt
        pred, hx = model(b.frames, dt=dt, hx=None if sever else hx)
        hx = hx.detach() if torch.is_tensor(hx) else hx
        P.append(pred.squeeze(-1).squeeze(0).float().cpu())
        Y.append(b.y.squeeze(0).float().cpu())
        V.append(b.valid.squeeze(0).cpu())
    return (data.to_degrees(torch.cat(P)).numpy(),
            data.to_degrees(torch.cat(Y)).numpy(),
            torch.cat(V).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=pathlib.Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=1,
                    help="1 = sever before every frame (the real test)")
    a = ap.parse_args()

    cfg = json.loads((a.run / "config.json").read_text())
    data = SteeringData(a.processed, device=a.device)
    model = build_arm(cfg["arm"], dropout=cfg.get("dropout", 0.0),
                      feature_norm=cfg.get("feature_norm", False)).to(a.device)
    ck = torch.load(a.run / "checkpoints" / "best.pt", map_location=a.device)
    model.load_state_dict(ck["model"])
    fdt = cfg.get("fixed_dt", False)

    out = {"run": a.run.name, "arm": cfg["arm"], "seed": cfg.get("seed"), "split": a.split}
    for tag, sever, chunk in (("carried", False, 256), ("severed", True, a.chunk)):
        p, y, v = rollout(model, data, a.split, fdt, sever, chunk)
        out[tag] = {"macro_skill": metrics.macro_skill(p, y, v),
                    "macro_mae": metrics.macro_mae(p, y, v),
                    "global_mae": float(np.abs(p[v] - y[v]).mean()),
                    "pearson_r": metrics.pearson_r(p, y, v)}
        out[f"_pred_{tag}"] = p

    pc, ps = out.pop("_pred_carried"), out.pop("_pred_severed")
    out["agreement_r"] = float(np.corrcoef(pc, ps)[0, 1])
    out["mean_abs_diff_deg"] = float(np.abs(pc - ps).mean())
    out["state_contribution_deg"] = out["severed"]["macro_mae"] - out["carried"]["macro_mae"]

    (a.run / f"state_ablation_{a.split}.json").write_text(json.dumps(out, indent=2) + "\n")
    c, s = out["carried"], out["severed"]
    print(f"\n{a.run.name}  [{a.split}]  arm={cfg['arm']}")
    print(f"  {'':<10}{'macroMAE':>10}{'macroSkill':>12}{'r':>9}")
    print(f"  {'carried':<10}{c['macro_mae']:>10.2f}{c['macro_skill']:>+12.4f}{c['pearson_r']:>+9.3f}")
    print(f"  {'severed':<10}{s['macro_mae']:>10.2f}{s['macro_skill']:>+12.4f}{s['pearson_r']:>+9.3f}")
    print(f"  状态贡献 {out['state_contribution_deg']:+.3f} 度   "
          f"两组预测一致性 r={out['agreement_r']:.5f}   平均差 {out['mean_abs_diff_deg']:.4f} 度")


if __name__ == "__main__":
    main()
