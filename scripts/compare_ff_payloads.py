"""Compare two fake-factor payload directories correction by correction (v7 spec 6.8: the coarse measurement of each
2016 half against the combined 2016 one; also one variant against another): every correction evaluated at the
centres of the bins of both payloads (and every category key), the nominal; prints the number of points, the
maximum and mean relative difference of the second to the first and the points with a key of only one of them (the
Wjets fractions of the W variant).

    python scripts/compare_ff_payloads.py <output_dir>/fake_factors/2016 <output_dir>/fake_factors/2016preVFP [--channels et,mt,tt]
"""
import argparse
from pathlib import Path

from shapesmith.payloads import compare

FILES = ("fake_factors", "FF_corrections")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--channels", default="et,mt,tt")
    args = parser.parse_args()
    print(f"{'channel':>7} {'correction':<55} {'points':>6} {'max |2/1-1|':>12} {'mean':>8} {'unmatched':>9}")
    for channel in args.channels.split(","):
        for name in FILES:
            first, second = (directory / f"{name}_{channel}.json.gz" for directory in (args.first, args.second))
            if not first.exists() or not second.exists():
                print(f"{channel:>7} {name}: missing in {args.first if not first.exists() else args.second}")
                continue
            for correction, entry in compare(first, second, {"syst": "nominal"}).items():
                print(f"{channel:>7} {correction:<55} {entry['points']:>6} {entry['max']:>12.4f} {entry['mean']:>8.4f} {entry.get('unmatched', 0):>9}")


if __name__ == "__main__":
    main()
