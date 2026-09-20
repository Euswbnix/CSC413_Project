"""Guards the README's rubric accounting.

`scripts/readme_status.py` is only trustworthy if the markers it parses are complete and
well-formed. Without this test, editing one marker silently breaks the point accounting and
the freeze-date check passes on a README that is missing a section.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
MARKER = re.compile(
    r"<!-- RUBRIC: (?P<name>.+?) \| (?P<points>\d+) \| (?P<group>\w+) \| "
    r"owner: (?P<owner>.*?) \| unblocked-by: (?P<unblocked>.*?) -->")

# The marking scheme, verbatim. If the handout changes, change it HERE and let the test
# tell you which README sections are missing.
RUBRIC = {
    "Introduction": 4, "Model Figure": 4, "Model Parameters": 4, "Model Examples": 4,
    "Data Source": 1, "Data Summary": 4, "Data Transformation": 3, "Data Split": 2,
    "Training Curve": 4, "Hyperparameter Tuning": 4, "Quantitative Measures": 2,
    "Quantitative and Qualitative Results": 8, "Justification of Results": 20,
    "Ethical Consideration": 4, "Authors": 2,
}
GROUP_TOTALS = {"readme": 70, "advanced": 10, "code": 20}


@pytest.fixture(scope="module")
def markers():
    found = [m.groupdict() for m in MARKER.finditer(README.read_text())]
    assert found, "no RUBRIC markers in README.md -- readme_status.py would report nothing"
    return found


def test_every_scored_readme_item_has_a_section(markers):
    have = {m["name"]: int(m["points"]) for m in markers if m["group"] == "readme"}
    missing = set(RUBRIC) - set(have)
    assert not missing, f"README has no section for: {sorted(missing)}"
    extra = set(have) - set(RUBRIC)
    assert not extra, f"marked as a scored 70-point item but not in the rubric: {sorted(extra)}"
    wrong = {k: (have[k], RUBRIC[k]) for k in RUBRIC if have[k] != RUBRIC[k]}
    assert not wrong, f"point value disagrees with the rubric (have, want): {wrong}"


def test_group_totals_are_the_marking_scheme(markers):
    totals = {}
    for m in markers:
        totals[m["group"]] = totals.get(m["group"], 0) + int(m["points"])
    assert totals == GROUP_TOTALS, totals
    assert sum(totals.values()) == 100


def test_section_order_follows_the_rubric(markers):
    """The formatting criterion rewards a reader being able to find each item quickly, and
    rubric order is the order a grader reads in."""
    order = [m["name"] for m in markers if m["group"] == "readme"]
    assert order == list(RUBRIC), f"sections out of rubric order:\n{order}\n{list(RUBRIC)}"


def test_the_authors_no_real_vehicle_request_is_at_the_top(markers):
    """Passing the restriction downstream rather than treating it as discharged by our own
    compliance. It must be above the first heading, not filed under Ethics."""
    text = README.read_text()
    head = text[:text.index("\n## ")]
    assert "never be used to drive a car" in head
    assert "MIT" in head and "licence" in head, (
        "the top block must state the gap between what the licence permits and what the "
        "author asks -- that gap IS the honest answer, not a footnote"
    )


def test_no_unfilled_placeholder_brackets(markers):
    """Catches `[X.X]`-style stubs left in prose. TODO blocks are fine and tracked; a bare
    placeholder inside a written sentence is what ships by accident."""
    bad = re.findall(r"\b[XY]\.[XY]\b|\bTBD°|\[FILL[^\]]*\]", README.read_text())
    assert not bad, f"unfilled placeholders in prose: {set(bad)}"
