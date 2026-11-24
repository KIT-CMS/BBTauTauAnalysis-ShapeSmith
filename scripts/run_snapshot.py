"""The content of the Analysis of every run YAML (tests/snapshot.py): pin it, dump it or compare two dumps.

    python scripts/run_snapshot.py --write                       # digests -> tests/fixtures/run_snapshot.json
    python scripts/run_snapshot.py --dump runs.json              # the full content of every run
    python scripts/run_snapshot.py --compare old.json new.json   # where two dumps differ
    python scripts/run_snapshot.py                               # the runs whose digest differs from the pinned one

tests/test_run_snapshot.py checks the digests. To see what moved a run: --dump on the state before (e.g. a
`git archive` of it, with PYTHONPATH on its src directories and this script run from its copy of the repository) and on
the state after, then --compare.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.snapshot import FIXTURE, differences, digest, snapshot  # noqa: E402 (the repository root comes first)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="write the digests of every run to the fixture")
    parser.add_argument("--state", default="", help="with --write: what the digests pin (e.g. the commits), recorded in the fixture")
    parser.add_argument("--dump", type=Path, help="write the full content of every run here")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("OLD", "NEW"), help="print the differences of two dumps")
    args = parser.parse_args()
    if args.compare:
        old, new = (json.loads(path.read_text()) for path in args.compare)
        changed = {run: differences(old.get(run, "<absent>"), new.get(run, "<absent>")) for run in sorted(set(old) | set(new))}
        changed = {run: lines for run, lines in changed.items() if lines}
        for run, lines in changed.items():
            print(f"{run}: {len(lines)} differences")
            print("\n".join(f"  {line}" for line in lines[:40]) + (f"\n  ... {len(lines) - 40} more" if len(lines) > 40 else ""))
        print(f"{len(changed)} of {len(set(old) | set(new))} runs differ")
        sys.exit(1 if changed else 0)
    contents = snapshot()
    digests = {run: digest(content) for run, content in contents.items()}
    if args.dump:
        args.dump.write_text(json.dumps(contents, indent=1, sort_keys=True))
        print(f"{args.dump}: {len(contents)} runs, {sum('error' in content for content in contents.values())} of them fail to build")
    if args.write:
        FIXTURE.write_text(json.dumps({"state": args.state, "runs": digests}, indent=1, sort_keys=True) + "\n")
        print(f"{FIXTURE}: {len(digests)} digests")
    if not args.dump and not args.write:
        pinned = json.loads(FIXTURE.read_text())["runs"]
        moved = sorted(run for run in set(pinned) | set(digests) if pinned.get(run) != digests.get(run))
        print("\n".join(moved) if moved else f"all {len(digests)} runs as pinned")
        sys.exit(1 if moved else 0)


if __name__ == "__main__":
    main()
