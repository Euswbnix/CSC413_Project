"""SwanLab experiment tracking, shared by every training script.

Tracking never changes or stops training. If swanlab is missing, not logged in, or the network
fails, the run carries on untracked (or offline) and says so once. SwanLab draws run colours and
ids from Python's global `random`, so that state is saved and restored around every call; no
training code here uses it, but a script that did would otherwise see a different stream.

Modes (--swanlab, default $CSC413_SWANLAB, else auto):
  auto     cloud if an API key is found (~/.swanlab/.netrc or $SWANLAB_API_KEY), else offline
  cloud    upload to swanlab.cn; a run given a run_id is resumed, so a checkpointed run that is
           restarted keeps one continuous chart
  offline  keep the log under $SWANLAB_LOG_DIR only; upload later with `swanlab sync <dir>`
  off      do not import swanlab at all (always the case under pytest)
Logs go to $SWANLAB_LOG_DIR (hdd/env.sh points it into ~/workspace), else ./swanlog.
$CSC413_SWANLAB_PROJECT replaces the script's project name (e.g. for smoke tests).
"""
import os
import random
import secrets
import sys

MODES = ("auto", "cloud", "offline", "off")


def default_mode():
    if "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules:
        return "off"
    return os.environ.get("CSC413_SWANLAB", "auto")


def add_argument(ap):
    ap.add_argument("--swanlab", choices=MODES, default=default_mode(),
                    help="experiment tracking (see tracking.py); default $CSC413_SWANLAB or auto")


def new_run_id(name):
    """A fresh id for a run that has no checkpoint yet. Store it in the checkpoint so a restart
    resumes the same SwanLab run; uses os.urandom, never the global random streams."""
    return f"{name}-{secrets.token_hex(4)}"[-64:]


def _say(msg):
    print(f"[swanlab] {msg}", flush=True)


def _quiet_nvml_shutdown(prev):
    """swanlab 0.7.15's GpuCollector.__del__ calls nvmlShutdown once too often; Python then prints
    an 'Exception ignored' traceback into the training log that looks like a crash. Drop that one
    case and pass everything else on."""
    def hook(u):
        if (type(u.exc_value).__name__.startswith("NVMLError")
                and "GpuCollector" in getattr(u.object, "__qualname__", "")):
            return
        prev(u)
    return hook


class _Guard:
    """Keeps Python's global random state unchanged across a SwanLab call."""

    def __enter__(self):
        self.state = random.getstate()

    def __exit__(self, *exc):
        random.setstate(self.state)
        return False


class Off:
    mode = "off"
    run_id = None

    def log(self, metrics, step=None):
        pass

    def finish(self):
        pass


class Run:
    def __init__(self, swanlab, mode, run_id):
        self._sl, self.mode, self.run_id, self._failed = swanlab, mode, run_id, False

    def log(self, metrics, step=None):
        if self._failed:
            return
        try:
            with _Guard():
                self._sl.log({k: float(v) for k, v in metrics.items() if v is not None}, step=step)
        except Exception as e:                          # never let logging stop training
            self._failed = True
            _say(f"log failed ({type(e).__name__}: {e}); the rest of this run is untracked")

    def finish(self):
        try:
            with _Guard():
                self._sl.finish()
        except Exception as e:
            _say(f"finish failed ({type(e).__name__}: {e})")


def _has_key():
    if os.environ.get("SWANLAB_API_KEY"):
        return True
    try:
        from swanlab.package import get_key
        get_key()
        return True
    except Exception:
        return False


def start(mode, project, name, config=None, group=None, tags=None, run_id=None):
    """Open a run and return an object with .log(metrics, step) and .finish().
    Call finish() before starting the next run in the same process."""
    if mode == "off":
        return Off()
    try:
        import swanlab
    except Exception as e:
        _say(f"not available ({type(e).__name__}: {e}); training continues untracked")
        return Off()
    if not getattr(sys.unraisablehook, "_csc413", False):
        sys.unraisablehook = _quiet_nvml_shutdown(sys.unraisablehook)
        sys.unraisablehook._csc413 = True
    if mode == "auto":
        mode = "cloud" if _has_key() else "offline"
        if mode == "offline":
            _say("no API key found: logging offline (swanlab login, then swanlab sync, to upload)")
    elif mode == "cloud" and not _has_key():
        # with a terminal attached swanlab would stop and ask for a key; never block a run on that
        _say("--swanlab cloud but no API key found: logging offline instead")
        mode = "offline"
    project = os.environ.get("CSC413_SWANLAB_PROJECT") or project
    kw = dict(project=project, experiment_name=name, config=config or {}, group=group, tags=tags,
              mode=mode)
    if mode == "cloud" and run_id:
        kw.update(id=run_id, resume="allow")              # resume is cloud-only in swanlab
    try:
        with _Guard():
            swanlab.init(**kw)
    except Exception as e:
        _say(f"init failed ({type(e).__name__}: {e}); training continues untracked")
        return Off()
    return Run(swanlab, mode, run_id)
