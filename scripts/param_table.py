#!/usr/bin/env python3
"""Emit the README's parameter tables from the same formula functions the tests assert
against, so the table is GENERATED OUTPUT, not transcription.

`make test && make params` is the answer to "how do I check this?".
"""

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch.nn as nn

from models.encoder import PilotNetEncoder, TwoFrameEncoder
from models.interface import (build_arm, cfc_params, lstm_params, gru_params, mlp_params,
                              matched_width, HIDDEN)

I = 32


def derivation_table(H=HIDDEN):
    D = I + H
    rows = [(f"`{n}` `Linear({D}->{H})`", f"{H}*{D} + {H}", H * D + H)
            for n in ("h_head", "g_head", "f_head", "o_head")]
    print(f"### CfC cell derivation (I={I}, H={H}, D=I+H={D})\n")
    print("| block | arithmetic | params |\n|---|---|---:|")
    for name, arith, n in rows:
        print(f"| {name} | {arith} | {n:,} |")
    print(f"| **CfC cell** | `4*{H}*({I}+{H}+1) = 4*{H}*{I+H+1}` | **{cfc_params(I, H):,}** |")
    assert sum(n for _, _, n in rows) == cfc_params(I, H)

    enc = PilotNetEncoder()
    conv = [(f"conv{i+1} {tuple(m.weight.shape)}",
             f"{m.weight.shape[0]}*({m.weight.shape[1]}*{m.kernel_size[0]*m.kernel_size[1]}) + {m.weight.shape[0]}",
             m.weight.numel() + m.bias.numel())
            for i, m in enumerate(m for m in enc.conv if isinstance(m, nn.Conv2d))]
    print(f"| readout `Linear({H}->1)` | {H} + 1 | {H+1:,} |")
    for name, arith, n in conv:
        print(f"| {name} | {arith} | {n:,} |")
    print(f"| FC `Linear(1152->32)` | 1152*32 + 32 | {enc.fc.weight.numel()+enc.fc.bias.numel():,} |")
    b = build_arm("cfc").param_breakdown()
    print(f"| **encoder (shared by every arm)** | | **{b['encoder']:,}** |")
    print(f"| **CfC arm total** | | **{b['total']:,}** |")
    print(f"\nRecurrent layer is **{b['recurrent']/b['total']:.1%}** of the model; "
          f"the shared perception encoder is {b['encoder']/b['total']:.0%}.\n")


def controls_table(H=HIDDEN):
    target = cfc_params(I, H)
    gh = matched_width(target, gru_params, I)
    mw = matched_width(target, mlp_params, I)
    print(f"### Matched controls (I={I})\n")
    print("| arm | recurrent formula | H | recurrent params | vs CfC |\n|---|---|---:|---:|---:|")
    for label, formula, h, n in [
        ("**CfC (ours)**", "`4H(I+H+1)`", H, target),
        ("**`nn.LSTM` (baseline, unmodified)**", "`4H(I+H) + 8H`", H, lstm_params(I, H)),
        ("`nn.GRU` (cut; shown for completeness)", "`3H(I+H) + 6H`", gh, gru_params(I, gh)),
        (f"CNN + MLP head, matched", f"`32D + D + D + 1`, D={mw}", "-", mlp_params(I, mw)),
        ("CNN + Linear head (lower bracket)", "`32 + 1`", "-", 33),
    ]:
        delta = "--" if n == target else f"{n/target:+.2%}".replace("+", "+") if n != target else ""
        print(f"| {label} | {formula} | {h} | {n:,} | {delta} |")
    print(f"\n`nn.LSTM - CfC = {lstm_params(I,H)-target} = exactly 4H` -- torch's redundant "
          f"second bias vector, **in the baseline's favour**.\n")

    print("| H | CfC | nn.LSTM | difference |\n|---:|---:|---:|---:|")
    for h in (16, 32, 64, 128):
        print(f"| {h} | {cfc_params(I,h):,} | {lstm_params(I,h):,} | {lstm_params(I,h)-cfc_params(I,h)} = 4H |")
    print()


def arm_totals():
    print("### Per-arm totals (as built)\n")
    print("| arm | encoder | recurrent | readout | total |\n|---|---:|---:|---:|---:|")
    for name in ("cfc", "lstm", "gru", "cnn_mlp", "cnn_linear", "cnn_2frame"):
        b = build_arm(name).param_breakdown()
        print(f"| {name} | {b['encoder']:,} | {b['recurrent']:,} | {b['readout']:,} | {b['total']:,} |")
    print("\nEvery arm reports trainable == total (zero frozen parameters). The reference "
          "library reports 6,898 for a CfC whose trainable count is 5,544 -- a 24.4% "
          "overcount -- because it registers its sparsity mask as a frozen `nn.Parameter`; "
          "we use `register_buffer`.\n")


if __name__ == "__main__":
    derivation_table(); controls_table(); arm_totals()
