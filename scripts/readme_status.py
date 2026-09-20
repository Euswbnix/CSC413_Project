#!/usr/bin/env python3
"""Point-weighted README completion. This is PROJECT.md section 12's checklist, as a command.

Every scored rubric item in README.md carries a machine-readable marker:

    <!-- RUBRIC: <name> | <points> | <group> | owner: <who> | unblocked-by: <what> -->

A section counts as WRITTEN when its body contains no TODO marker. The useful output is not
the total -- it is the WRITABLE NOW list: items still unwritten whose `unblocked-by` is
`nothing`. Those need no experiment, no data and no GPU, and they are the first casualty of
saving the writing for the final week.

    python scripts/readme_status.py
    python scripts/readme_status.py --fail-if-incomplete    # for the freeze date / CI
"""

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


import argparse
import pathlib
import re
import sys

MARKER = re.compile(
    r"<!-- RUBRIC: (?P<name>.+?) \| (?P<points>\d+) \| (?P<group>\w+) \| "
    r"owner: (?P<owner>.*?) \| unblocked-by: (?P<unblocked>.*?) -->")
HEADING = re.compile(r"^#{1,6} ", re.M)
TODO = re.compile(r"\bTODO\b")   # bold prose TODO or an inline `# TODO`
GROUP_TOTALS = {"readme": 70, "advanced": 10, "code": 20}


def sections(text):
    hits = list(MARKER.finditer(text))
    for i, m in enumerate(hits):
        start = m.end()
        stop = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        nxt = HEADING.search(text, start, stop)          # body ends at the next heading
        body = text[start:nxt.start() if nxt else stop]
        yield m.groupdict(), body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", type=pathlib.Path, default=pathlib.Path("README.md"))
    ap.add_argument("--fail-if-incomplete", action="store_true",
                    help="exit 1 if any scored item is still TODO. Use from the freeze date.")
    args = ap.parse_args()

    text = args.readme.read_text()
    rows = [(d, not TODO.search(body)) for d, body in sections(text)]
    if not rows:
        raise SystemExit(f"no RUBRIC markers found in {args.readme}")

    width = max(len(d["name"]) for d, _ in rows)
    print(f"{'item':{width}}  {'pts':>4}  {'status':7}  {'owner':10}  unblocked-by")
    print("-" * (width + 40))
    done = {}
    for d, written in sorted(rows, key=lambda r: (-int(r[0]["points"]), r[0]["name"])):
        g, pts = d["group"], int(d["points"])
        done.setdefault(g, 0)
        if written:
            done[g] += pts
        flag = "WRITTEN" if written else "todo"
        print(f"{d['name']:{width}}  {pts:>4}  {flag:7}  {d['owner']:10}  {d['unblocked']}")

    print()
    earned = sum(done.values())
    for g, total in GROUP_TOTALS.items():
        got = done.get(g, 0)
        bar = "#" * round(20 * got / total) + "." * (20 - round(20 * got / total))
        print(f"  {g:9} {bar} {got:>3}/{total} pts drafted")
    print(f"  {'TOTAL':9} {'':20} {earned:>3}/{sum(GROUP_TOTALS.values())} pts drafted")

    writable = [d for d, written in rows
                if not written and d["unblocked"].strip().lower() in ("nothing", "none")]
    if writable:
        n = sum(int(d["points"]) for d in writable)
        print(f"\nWRITABLE NOW -- {n} points needing no experiment, no data and no GPU:")
        for d in sorted(writable, key=lambda d: -int(d["points"])):
            print(f"  {d['points']:>3} pts  {d['name']}  (owner: {d['owner']})")
        print("  These are the first casualty of saving the writing for the final week.")

    unnamed = [d for d, _ in rows if "TBD" in d["owner"]]
    if unnamed:
        print(f"\nUNASSIGNED OWNER on {sum(int(d['points']) for d in unnamed)} points: "
              + ", ".join(d["name"] for d in unnamed))
        print("  Justification is 20 points on its own -- it needs a NAMED person, not 'the "
              "one who writes things'.")

    blocked = sorted({d["unblocked"] for d, written in rows
                      if not written and d["unblocked"].strip().lower() not in ("nothing", "none")})
    if blocked:
        print("\nBLOCKED ON:")
        for b in blocked:
            pts = sum(int(d["points"]) for d, w in rows if not w and d["unblocked"] == b)
            print(f"  {pts:>3} pts  <- {b}")

    if args.fail_if_incomplete and earned < sum(GROUP_TOTALS.values()):
        print("\nFAIL: scored sections are still TODO.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
