"""D4: the pre-registered comparison, one arm and one stage per process so seeds run in parallel.

Everything here is fixed by docs/preregistration_2026-09-22.md: mixed-rate training, one
checkpoint evaluated under every condition, the {1e-3, 3e-4, 1e-4, 3e-5} grid at one seed
followed by eight seeds at the winner, 60 epochs, patience 8, five mask draws, and selection by
the mean macro MAE over the three conditions. Validation only; the test split is read by
--split-name test, which the pre-registration allows exactly once, at the end.

Features come from the shared cache (hdd/make_cache.py) through mmap, so running six of these at
once costs one copy of the data, not six. Every epoch is checkpointed under <out>/ckpt/ and a
rerun of the same command resumes where it stopped, so a shutdown costs at most one epoch.

    python hdd/d4_run.py --cache ~/data/hdd/cache/dinov2_10hz --arm cfc --stage lr  --out ~/data/hdd/d4
    python hdd/d4_run.py --cache ~/data/hdd/cache/dinov2_10hz --arm cfc --stage final --seeds 0 1 --out ~/data/hdd/d4
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import metrics
from models.cfc import CfC

MOVING = 3.0
MIN_DT = 1e-3          # never feed dt = 0 to the LTC solver


def refuse_inside_git(path):
    d = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path) or ".")
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            sys.exit(f"refusing to write HDD-derived output inside a git working tree ({d})")
        parent = os.path.dirname(d)
        if parent == d:
            return
        d = parent


class Cached:
    """Windows over the mmapped base-rate cache. A window is valid only if its target is a target
    frame and no step inside it is longer than `max_step` (the D3 filter only capped the span)."""

    def __init__(self, cache, which, steps, device, max_step=0.2):
        idx = json.load(open(os.path.join(cache, "index.json")))
        self.steps, self.device, self.items, counts = steps, device, [], []
        for s, meta in sorted(idx["sessions"].items()):
            if meta["split"] != which:
                continue
            f = np.load(os.path.join(cache, f"{s}.feats.npy"), mmap_mode="r")
            m = np.load(os.path.join(cache, f"{s}.meta.npy"), mmap_mode="r")
            t, steer, speed = m[:, 0], m[:, 1], m[:, 2]
            ok = np.isfinite(steer) & np.isfinite(speed) & (speed > MOVING)
            tgt = np.flatnonzero(ok)
            tgt = tgt[tgt >= steps - 1]
            if not len(tgt):
                continue
            win = tgt[:, None] - np.arange(steps - 1, -1, -1)[None]
            keep = np.diff(t[win], axis=1).max(1) <= max_step
            tgt = tgt[keep]
            if not len(tgt):
                continue
            self.items.append(dict(session=s, cluster=meta["cluster"], feats=f, t=t,
                                   y=steer.astype(np.float32), tgt=tgt))
            counts.append(len(tgt))
        self.n = int(sum(counts))

    def labels(self):
        return np.concatenate([it["y"][it["tgt"]] for it in self.items]).astype(np.float64)

    def clusters(self):
        return np.concatenate([[it["cluster"]] * len(it["tgt"]) for it in self.items])

    def sessions(self):
        return np.concatenate([[it["session"]] * len(it["tgt"]) for it in self.items])

    def batches(self, batch, shuffle, rng=None):
        order = np.arange(len(self.items))
        if shuffle:
            rng.shuffle(order)
        for i in order:
            it = self.items[i]
            tgt = it["tgt"]
            if shuffle:
                tgt = tgt[rng.permutation(len(tgt))]
            for j in range(0, len(tgt), batch):
                b = tgt[j:j + batch]
                win = b[:, None] - np.arange(self.steps - 1, -1, -1)[None]
                f = torch.from_numpy(np.ascontiguousarray(it["feats"][win])).to(self.device).float()
                t = torch.from_numpy(it["t"][win]).to(self.device)
                y = torch.from_numpy(it["y"][b]).to(self.device)
                yield f, t, y


class LTCBlock(nn.Module):
    """ncps LTC driven one step at a time, with dt passed per sample as (B, 1).

    Stepping manually is what avoids the known defect where a batch equal to the hidden size makes
    `timespans[:, t].squeeze()` broadcast one sample's elapsed time onto another (ncps 1.0.1,
    PR #85). The reversal potentials are seeded per run instead of the library's fixed 1111."""

    def __init__(self, hidden, seed, ode_unfolds=6):
        super().__init__()
        from ncps.torch.ltc_cell import LTCCell
        from ncps.wirings import FullyConnected
        self.cell = LTCCell(FullyConnected(hidden, erev_init_seed=seed), in_features=hidden,
                            ode_unfolds=ode_unfolds)
        self.hidden = hidden

    def forward(self, x, dt):
        h = x.new_zeros(x.shape[0], self.hidden)
        out = []
        for i in range(x.shape[1]):
            h, _ = self.cell(x[:, i], h, dt[:, i:i + 1].clamp(min=MIN_DT))
            out.append(h)
        return torch.stack(out, 1)


class Arm(nn.Module):
    """Shared projection and readout; the recurrent block is the only difference."""

    def __init__(self, kind, dim, hidden=64, seed=0, ode_unfolds=6):
        super().__init__()
        self.kind = kind
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(dim, hidden)
        self.rnn = (nn.LSTM(hidden + 1, hidden, batch_first=True) if kind == "lstm" else
                    CfC(hidden, hidden) if kind == "cfc" else LTCBlock(hidden, seed, ode_unfolds))
        self.head = nn.Linear(hidden, 1)

    def forward(self, f, dt, valid):
        x = self.proj(self.norm(f)) * valid[:, :, None]
        if self.kind == "lstm":
            out, _ = self.rnn(torch.cat([x, dt[:, :, None]], -1))
        elif self.kind == "cfc":
            out, _ = self.rnn(x, dt)
        else:
            out = self.rnn(x, dt)
        last = valid.float().cumsum(1).argmax(1)
        return self.head(out[torch.arange(len(out), device=out.device), last]).squeeze(-1)

    def blocks(self):
        n = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.rnn.parameters() if not p.requires_grad)
        return dict(projection=n(self.proj), temporal=n(self.rnn), readout=n(self.head),
                    temporal_frozen=frozen)


def masked(t, keep, gen):
    """Keep the current frame and a random `keep` share of the history; dt over what survives."""
    mask = torch.rand(t.shape, device=t.device, generator=gen) < keep if keep < 1 else \
        torch.ones_like(t, dtype=torch.bool)
    mask[:, -1] = True
    T = t.shape[1]
    order = torch.argsort(torch.where(mask, torch.arange(T, device=t.device).expand_as(t),
                                      torch.full_like(t, T + 1, dtype=torch.long)), dim=1)
    return mask, order


def forward_masked(model, f, t, keep, gen):
    mask, order = masked(t, keep, gen)
    n = int(mask.sum(1).max())
    idx = order[:, :n]
    kf = torch.gather(f, 1, idx[:, :, None].expand(-1, -1, f.shape[2]))
    kt = torch.gather(t, 1, idx)
    valid = torch.gather(mask, 1, idx)
    dt = torch.zeros_like(kt)
    dt[:, 1:] = (kt[:, 1:] - kt[:, :-1]).clamp(min=0)
    return model(kf, dt.to(kf.dtype), valid)


def predict(model, data, mu, sd, keep, mask_seed):
    model.eval()
    gen = torch.Generator(device=data.device).manual_seed(mask_seed)
    out = []
    with torch.no_grad():
        for f, t, y in data.batches(512, shuffle=False):
            out.append((forward_masked(model, f, t, keep, gen) * sd + mu).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def score_all(model, data, y, mu, sd, keeps, draws):
    """Mean macro MAE over the conditions, plus the per-condition numbers and predictions."""
    per, preds = {}, {}
    for keep in keeps:
        vals = []
        for d in range(draws if keep < 1 else 1):
            p = predict(model, data, mu, sd, keep, 1000 * d + 7)
            preds[f"k{keep:g}_m{d}"] = p
            vals.append(dict(mask=d, macro_mae=metrics.macro_mae(p, y), ccc=metrics.ccc(p, y)))
        per[f"keep_{keep:g}"] = vals
    mean = float(np.mean([np.mean([v["macro_mae"] for v in rows]) for rows in per.values()]))
    return mean, per, preds


def train(arm, train_d, val_d, y_val, lr, seed, args, mu, sd, dim, tag):
    """Trains with a checkpoint after every epoch: a run that is killed (or a machine that is shut
    down) resumes from the last finished epoch with the same model, optimiser and random streams,
    so the result is the one an uninterrupted run would have given."""
    torch.manual_seed(seed)
    model = Arm(arm, dim, seed=seed, ode_unfolds=args.ode_unfolds).to(train_d.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=train_d.device).manual_seed(seed)
    best, best_state, bad, start = np.inf, None, 0, 0
    ckpt = os.path.join(args.out, "ckpt", f"{arm}_lr{lr:g}_s{seed}.pt")
    if os.path.exists(ckpt):
        # load on the CPU: RNG states must stay CPU byte tensors, and load_state_dict moves the
        # weights and optimiser state onto the model's device by itself
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        rng.bit_generator.state = c["rng"]
        gen.set_state(c["gen"])
        torch.set_rng_state(c["torch_rng"])
        if torch.cuda.is_available() and c.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(c["cuda_rng"])
        best, best_state, bad, start = c["best"], c["best_state"], c["bad"], c["epoch"] + 1
        print(f"[{tag}] resumed lr {lr:g} seed {seed} at epoch {start} (best {best:.3f}, patience used {bad})")
        if bad >= args.patience or start >= args.epochs:
            model.load_state_dict(best_state)
            return model, best, start
    ep = start - 1
    for ep in range(start, args.epochs):
        model.train()
        for f, t, y in train_d.batches(args.batch, shuffle=True, rng=rng):
            keep = float(rng.choice(args.keep_rates))
            loss = ((forward_masked(model, f, t, keep, gen) - (y - mu) / sd) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        mean, _, _ = score_all(model, val_d, y_val, mu, sd, args.keep_rates, 1)
        if mean < best - 1e-6:
            best, bad = mean, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"[{tag}] lr {lr:g} seed {seed} epoch {ep + 1}: {mean:.3f} (best {best:.3f}, patience {bad}/{args.patience})")
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        tmp = ckpt + ".tmp"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), rng=rng.bit_generator.state,
                        gen=gen.get_state(), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        best=best, best_state=best_state, bad=bad, epoch=ep), tmp)
        os.replace(tmp, ckpt)                                   # never leave a half-written file
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best, ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--arm", required=True, choices=("lstm", "cfc", "ltc"))
    ap.add_argument("--stage", required=True, choices=("lr", "final"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-name", default="val", choices=("val", "test"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4, 3e-5])
    ap.add_argument("--lr", type=float, help="final stage: the rate chosen by the lr stage")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--keep-rates", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    ap.add_argument("--mask-draws", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ode-unfolds", type=int, default=6)
    ap.add_argument("--max-step", type=float, default=0.2)
    ap.add_argument("--tag", default="", help="suffix for the output files, so parallel runs do not collide")
    a = ap.parse_args()
    refuse_inside_git(a.out)
    os.makedirs(a.out, exist_ok=True)
    os.chmod(a.out, 0o700)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    train_d = Cached(a.cache, "train", a.steps, device, a.max_step)
    val_d = Cached(a.cache, a.split_name, a.steps, device, a.max_step)
    ytr, yva = train_d.labels(), val_d.labels()
    mu, sd = float(ytr.mean()), float(ytr.std())
    dim = train_d.items[0]["feats"].shape[1]
    tag = f"{a.arm}_{a.stage}" + (f"_s{'-'.join(map(str, a.seeds))}" if a.stage == "final" else "")
    print(f"[{tag}] train {len(ytr):,} windows, {a.split_name} {len(yva):,}; dim {dim}; device {device}")

    rows = []
    if a.stage == "lr":
        for lr in a.lrs:
            _, mean, eps = train(a.arm, train_d, val_d, yva, lr, a.seeds[0], a, mu, sd, dim, tag)
            rows.append(dict(lr=lr, seed=a.seeds[0], mean_macro_mae=mean, epochs=eps))
            print(f"[{tag}] lr {lr:g}: mean over conditions {mean:.3f} ({eps} epochs)")
        best = min(rows, key=lambda r: r["mean_macro_mae"])["lr"]
        print(f"[{tag}] chosen lr {best:g}")
        json.dump(dict(arm=a.arm, stage="lr", rows=rows, chosen_lr=best),
                  open(os.path.join(a.out, f"{a.arm}_lr{a.tag}.json"), "w"), indent=1)
    else:
        if not a.lr:
            sys.exit("--lr is required for the final stage")
        for seed in a.seeds:
            model, mean, eps = train(a.arm, train_d, val_d, yva, a.lr, seed, a, mu, sd, dim, tag)
            _, per, preds = score_all(model, val_d, yva, mu, sd, a.keep_rates, a.mask_draws)
            np.savez_compressed(os.path.join(a.out, f"pred_{a.arm}_s{seed}_{a.split_name}.npz"),
                                y=yva.astype(np.float32), cluster=val_d.clusters(),
                                session=val_d.sessions(), **preds)
            row = dict(arm=a.arm, seed=seed, lr=a.lr, epochs=eps, mean_macro_mae=mean,
                       blocks=model.blocks(), conditions=per)
            json.dump(row, open(os.path.join(a.out, f"{a.arm}_s{seed}_{a.split_name}.json"), "w"), indent=1)
            rows.append(row)
            summary = " ".join(f"{k} {np.mean([v['macro_mae'] for v in rows_]):.3f}"
                               for k, rows_ in per.items())
            print(f"[{tag}] seed {seed} ({eps} epochs): {summary}")
    for f in os.listdir(a.out):
        if os.path.isfile(os.path.join(a.out, f)):
            os.chmod(os.path.join(a.out, f), 0o600)
    print(f"[{tag}] done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
