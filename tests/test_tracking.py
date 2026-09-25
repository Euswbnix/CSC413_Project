"""tracking.py must never change training: off under pytest, a no-op when off, and a SwanLab run
must leave every random stream exactly where it was."""
import os
import random

import numpy as np
import pytest
import torch

import tracking


def test_off_by_default_under_pytest():
    assert tracking.default_mode() == "off"


def test_off_is_a_no_op():
    run = tracking.start("off", "unused", "unused")
    assert isinstance(run, tracking.Off)
    run.log({"x": 1.0}, step=1)
    run.finish()


def test_guard_restores_python_random():
    random.seed(3)
    before = random.getstate()
    with tracking._Guard():
        random.random()
    assert random.getstate() == before


def test_run_id_is_stable_length_and_leaves_random_alone():
    random.seed(5)
    before = random.getstate()
    rid = tracking.new_run_id("ltc_lr0.001_s0_u24")
    assert random.getstate() == before
    assert rid.startswith("ltc_lr0.001_s0_u24-") and len(rid) <= 64
    assert not set("/\\#?%:") & set(rid)


def test_offline_run_keeps_every_random_stream(tmp_path, monkeypatch):
    pytest.importorskip("swanlab")
    monkeypatch.setenv("SWANLAB_LOG_DIR", str(tmp_path))
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    states = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state().clone())
    run = tracking.start("offline", "csc413-selftest", "rng_check", config={"a": 1})
    assert isinstance(run, tracking.Run)
    for step in (1, 2, 3):
        run.log({"val/x": step * 0.5, "none_is_skipped": None}, step=step)
    run.finish()
    assert random.getstate() == states[0]
    assert np.array_equal(np.random.get_state()[1], states[1])
    assert torch.equal(torch.get_rng_state(), states[2])
    assert any(p.name.startswith("run-") for p in tmp_path.iterdir())


def test_only_swanlabs_nvml_destructor_noise_is_silenced():
    from types import SimpleNamespace
    seen = []
    hook = tracking._quiet_nvml_shutdown(seen.append)

    class NVMLError_Uninitialized(Exception):
        pass

    def __del__():
        pass
    __del__.__qualname__ = "GpuCollector.__del__"
    hook(SimpleNamespace(exc_value=NVMLError_Uninitialized(), object=__del__))
    assert seen == []
    other = SimpleNamespace(exc_value=ValueError("x"), object=__del__)
    hook(other)
    assert seen == [other]
