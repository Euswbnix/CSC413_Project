"""Makes "ncps is a test-only dependency" a FACT rather than a promise."""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_SURFACE = ["models", "data", "scripts"]


def test_no_liquid_network_library_in_the_model_or_pipeline():
    offenders = []
    for d in MODEL_SURFACE:
        for f in (ROOT / d).rglob("*.py"):
            text = f.read_text()
            for i, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue                       # prose about ncps is fine; imports are not
                if "import ncps" in stripped or "from ncps" in stripped:
                    offenders.append(f"{f.relative_to(ROOT)}:{i}: {stripped}")
    assert not offenders, "ncps must never be imported outside tests/:\n" + "\n".join(offenders)


def test_arms_are_interchangeable_behind_one_interface():
    """The escape hatch, mechanised: every arm takes the same input and returns the same
    shape, so models/cfc.py's cell can be swapped for the library's without re-running a
    single experiment."""
    import torch
    from models.interface import build_arm, ARMS

    frames = torch.zeros(2, 3, 3, 66, 200)
    dt = torch.ones(2, 3)
    for name in ARMS:
        y, _ = build_arm(name)(frames, dt=dt)
        assert y.shape == (2, 3, 1), f"{name} returned {tuple(y.shape)}"
