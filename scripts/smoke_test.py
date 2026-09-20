#!/usr/bin/env python3
"""Day-zero gate. Run this BEFORE downloading 3.1 GB of data, and make it the first
command in the README's reproduction sequence.

It ASSERTS rather than prints, because the failure this exists to catch is silent: on an
RTX 5090 (Blackwell, sm_120) with driver 570.x-579.x, torch installs cleanly,
`torch.cuda.is_available()` returns True, the card's name is reported correctly -- and
then every kernel fails. A real on-device backward pass is the only check that sees it.

Usage:  python scripts/smoke_test.py [--allow-cpu]
Exit 0 = GO.
"""

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import importlib.metadata
import sys

import torch

from models.cfc import CfC
from models.interface import build_arm, cfc_params, lstm_params


def versions():
    print("=== environment")
    print(f"  torch  {torch.__version__}")
    for pkg in ("numpy", "ncps"):
        try:
            # importlib.metadata, NOT ncps.__version__ -- the 1.0.1 wheel's
            # ncps/__init__.py says __version__ = "0.0.2", so every run's config.json
            # would cite a version that does not exist on PyPI.
            print(f"  {pkg:6} {importlib.metadata.version(pkg)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  {pkg:6} not installed")
    if torch.version.cuda:
        print(f"  cuda   {torch.version.cuda}")
        print(f"  arch   {torch.cuda.get_arch_list()}")


def check_device(allow_cpu):
    print("=== device")
    if not torch.cuda.is_available():
        msg = ("  NO CUDA DEVICE. On the training machine this is a HARD FAILURE: "
               "read `nvidia-smi` and see requirements.txt for the driver/wheel matrix.")
        if not allow_cpu:
            print(msg)
            sys.exit(1)
        print(msg.replace("HARD FAILURE", "waived via --allow-cpu"))
        return torch.device("cpu")

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    print(f"  {name}  capability {arch}")
    assert arch in torch.cuda.get_arch_list(), (
        f"{arch} is NOT in this torch build's arch list {torch.cuda.get_arch_list()}. "
        "The card is detected but no kernels exist for it -- update the driver or install "
        "from the cu128 index (see requirements.txt)."
    )

    # TF32 OFF, explicitly. PyTorch's Blackwell defaults are ASYMMETRIC -- on for cuDNN
    # convolutions, off for cuBLAS matmuls -- so the PilotNet stack would run at a 10-bit
    # mantissa while the CfC's linears ran true fp32: an asymmetric precision difference
    # sitting inside the headline comparison. Costs a few ms/step; retires the question.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    print("  TF32 disabled (cudnn + matmul)")
    return torch.device("cuda")


def check_real_backward(device):
    """The assert that catches a detected-but-unusable GPU."""
    print("=== on-device forward + backward")
    m = build_arm("cfc").to(device)
    frames = torch.rand(2, 4, 3, 66, 200, device=device)
    dt = torch.ones(2, 4, device=device)
    y, _ = m(frames, dt=dt)
    assert y.shape == (2, 4, 1), f"expected (2,4,1); got {tuple(y.shape)}"
    y.pow(2).mean().backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    bad = [n for n, p in m.named_parameters()
           if p.requires_grad and (not torch.isfinite(p.grad).all() or p.grad.abs().max() == 0)]
    assert not bad, f"non-finite or all-zero gradient: {bad}"
    print(f"  output {tuple(y.shape)}; finite non-zero grads on all "
          f"{sum(1 for _ in m.parameters())} tensors")


def check_model_invariants():
    print("=== model invariants")
    m = build_arm("cfc")
    b = m.param_breakdown()
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert total == trainable, "frozen parameters are being counted as parameters"
    assert b == {"encoder": 168_244, "recurrent": 24_832, "readout": 65, "total": 193_141}, b
    assert cfc_params(32, 64) == 24_832
    assert lstm_params(32, 64) - cfc_params(32, 64) == 4 * 64
    print(f"  encoder {b['encoder']:,} + recurrent {b['recurrent']:,} + readout {b['readout']} "
          f"= {b['total']:,}  (recurrent {b['recurrent']/b['total']:.1%})")
    print(f"  trainable == total == {trainable:,}  (zero frozen)")

    # dt shape contract: closes the ncps bug class where B == hidden width broadcasts on
    # the wrong axis, silently, without raising.
    c = CfC(32, 64).cell
    for shape in [(8,), (8, 4), ()]:
        try:
            c(torch.zeros(8, 32), torch.zeros(8, 64), torch.ones(shape))
        except AssertionError:
            continue
        raise SystemExit(f"dt shape contract did not reject {shape}")
    print("  dt shape contract rejects (B,), (B,T) and scalar")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-cpu", action="store_true",
                    help="for a laptop. NEVER pass this on the training machine.")
    args = ap.parse_args()
    versions()
    device = check_device(args.allow_cpu)
    check_model_invariants()
    check_real_backward(device)
    print("\nGO")


if __name__ == "__main__":
    main()
