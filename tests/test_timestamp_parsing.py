"""Locks down the millisecond padding direction.

`YYYY-MM-DD HH:MM:SS:mmm` looks like something strptime can take with `%f`, and it cannot:
`%f` pads fractional seconds on the RIGHT, so ":11" parses as 110 ms instead of 011 ms. On
the 2018 release 9.9% of lines have a short ms field, and the wrong reading produced 3,574
apparently-backwards timestamps, a p1 dt of -750.8 ms, and 1,822 phantom recording gaps --
which would have cut the recording into 1,823 segments, most shorter than one training
window, and fed the CfC's time gate errors of up to +-900 ms. Nothing about it raises.
"""

from data.eda_sullychen import parse_timestamp

# Verbatim from data/raw/data.txt, rows 0-4 of the 2018 release. The ms sequence
# 912, 972, 11, 76, 105 is monotonic ONLY under left-padding.
REAL_ROWS = ["2018-07-01 17:09:44:912", "2018-07-01 17:09:44:972",
             "2018-07-01 17:09:45:11", "2018-07-01 17:09:45:76",
             "2018-07-01 17:09:45:105"]


def test_short_millisecond_fields_are_left_padded():
    assert parse_timestamp("2018-07-01 17:09:45:11").microsecond == 11_000
    assert parse_timestamp("2018-07-01 17:09:45:76").microsecond == 76_000
    assert parse_timestamp("2018-07-01 17:09:45:1").microsecond == 1_000
    assert parse_timestamp("2018-07-01 17:09:45:105").microsecond == 105_000


def test_real_consecutive_rows_are_monotonic():
    t = [parse_timestamp(s) for s in REAL_ROWS]
    assert all(b > a for a, b in zip(t, t[1:])), "timestamps went backwards"
    gaps = [(b - a).total_seconds() for a, b in zip(t, t[1:])]
    assert max(gaps) < 0.15, f"a plausible inter-frame gap became {max(gaps):.3f}s"


def test_the_wrong_reading_is_detectably_wrong():
    """Guards the guard: if strptime ever starts left-padding %f, this test tells us the
    hazard is gone rather than letting the workaround rot in place unexplained."""
    from datetime import datetime
    naive = datetime.strptime("2018-07-01 17:09:45:11", "%Y-%m-%d %H:%M:%S:%f")
    assert naive.microsecond == 110_000, (
        "%f no longer right-pads; re-derive parse_timestamp and update its comment")
