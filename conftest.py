"""Puts the repo root on sys.path so `pytest` and `python scripts/*.py` both work from
the repo root with no install step and no PYTHONPATH fiddling."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
