"""The per-era binning of the fake-factor tables (v7 spec 6.8), written to src/bbtautau_shapesmith/ff_edges.py, which
ff_tables reads: the reviewed bin counts (BIN_COUNTS) and the edges (PER_ERA_EDGES). Two steps, repeated until the
counts are stable:

    python scripts/ff_binning.py counts <output_dir>                 # review the bin counts on the measured records
    python scripts/ff_binning.py edges <measure log> --era 2017      # enter the edges of `measure --suggest-binning`

counts: the statistics rule on the records of the W-variant measurement of every era (<output_dir>/fake_factors_w/
<era>/measurement_<ch>.json; its quantities contain those of QCD + ttbar). For every equipopulated quantity, per
category, the median relative statistical error of the measured ratio (without the bins put in front by add_left, the
iso_1 == 0 bin): above 0.30 the count n becomes max(2, floor(n (0.25 / median)^2)) (spec: stat error per bin <= ~25-30
%); in 2016 and 2017 a median below 0.25 raises n to min(n_2018, floor(n (0.25 / median)^2)), at most the 2018 count.
The fractions keep theirs. A count equal to the current one is not recorded.

edges: the JSON lines "<channel> suggested edges (JSON): {...}" of the measure log (<output_dir>/logs/measure_<time>.log)
of a run with ff_processes [QCD, Wjets, ttbar]. --only keeps the quantities with one of the given prefixes,
`<prefix>` in every channel or `<channel>:<prefix>` in one (2018: the W+jets ones and the quantities with their own
2018 count or lower end; its other QCD and ttbar edges are the measured 2018 tables). The other eras are kept.
"""
from __future__ import annotations

import argparse
import json
import math
import pprint
import re
from pathlib import Path

import numpy as np

TARGET = Path(__file__).resolve().parents[1] / "src" / "bbtautau_shapesmith" / "ff_edges.py"
LINE = re.compile(r"(?P<channel>\w+) suggested edges \(JSON\): (?P<json>\{.*\})\s*$")
ERAS = ("2016", "2017", "2018")
LIMIT, TARGET_ERROR = 0.30, 0.25
HEADER = '''"""The per-era binning of the fake-factor tables (ff_tables, v7 spec 6.8), written by scripts/ff_binning.py; do not
edit. BIN_COUNTS: era -> channel -> quantity -> equipopulated bin count per category, where the statistics review of
the measured ratios changed it (over ff_tables.PER_ERA_OVERRIDES). PER_ERA_EDGES: era -> channel -> quantity -> edges
per category, from `shapesmith measure --suggest-binning` on the era's FF-measurement skim with those counts.

Sources:
{sources}
"""
'''


def read() -> tuple[dict, dict, dict]:
    if not TARGET.exists():
        return {}, {}, {}
    namespace: dict = {}
    exec(compile(TARGET.read_text(), str(TARGET), "exec"), namespace)
    return namespace.get("BIN_COUNTS", {}), namespace["PER_ERA_EDGES"], namespace.get("SOURCES", {})


def write(counts: dict, edges: dict, sources: dict) -> None:
    def ordered(d: dict) -> dict:
        return {era: d[era] for era in sorted(d)}

    text = HEADER.format(sources="\n".join(f"- {key}: {line}" for key, line in sorted(sources.items())))
    text += f"SOURCES = {pprint.pformat(dict(sorted(sources.items())), width=120, sort_dicts=False)}\n"
    text += f"BIN_COUNTS = {pprint.pformat(ordered(counts), width=120, sort_dicts=False, compact=True)}\n"
    text += f"PER_ERA_EDGES = {pprint.pformat(ordered(edges), width=120, sort_dicts=False, compact=True)}\n"
    TARGET.write_text(text)


def names(tables: dict) -> dict:
    from bbtautau_shapesmith import ff_tables

    result = {}
    for key, table in tables.items():
        result.update(ff_tables.quantity_names(key, table))
    return result


def record_name(name: str) -> str:
    """The record entry of a quantity: the orthogonal non-closures under for_DRtoSR/."""
    if "_for_DRtoSR_non_closure_" in name:
        return "for_DRtoSR/" + name.replace("_for_DRtoSR_non_closure_", "_non_closure_")
    return name


def review(output: Path, report: list[str]) -> dict:
    from bbtautau_shapesmith import ff_tables

    reference = {ch: names(t) for ch, t in ff_tables.tables("2018", True)[0].items()}
    result = {}
    for era in ERAS:
        tables, _ = ff_tables.tables(era, True)
        for channel, channel_tables in tables.items():
            record = json.loads((output / "fake_factors_w" / era / f"measurement_{channel}.json").read_text())
            for name, binned in names(channel_tables).items():
                if binned.equipopulated is None or name.startswith("process_fractions"):
                    continue
                entries = record[record_name(name)]
                skip = len(binned.equipopulated.add_left)
                counts = []
                for entry, n, n_2018 in zip(entries, binned.equipopulated.n_bins, reference[channel][name].equipopulated.n_bins):
                    y, errors = np.asarray(entry["y"])[skip:], np.asarray(entry["errors"])[skip:]
                    relative = np.where(y > 0, errors / np.where(y > 0, y, 1.0), np.inf)
                    median = float(np.median(relative))
                    new = n
                    if median > LIMIT and n > 2:
                        new = max(2, math.floor(n * (TARGET_ERROR / median) ** 2))
                    elif era != "2018" and median < TARGET_ERROR and n < n_2018:
                        new = min(n_2018, max(n, math.floor(n * (TARGET_ERROR / median) ** 2)))
                    counts.append(new)
                    report.append(f"{era} {channel} {name:45s} cat {entry['category']} n {n:2d} median {median:6.3f} -> {new}")
                if counts != list(binned.equipopulated.n_bins):
                    result.setdefault(era, {}).setdefault(channel, {})[name] = tuple(counts)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    counts_parser = sub.add_parser("counts")
    counts_parser.add_argument("output", type=Path)
    counts_parser.add_argument("--source", default=None)
    edges_parser = sub.add_parser("edges")
    edges_parser.add_argument("log", type=Path)
    edges_parser.add_argument("--era", required=True, choices=ERAS)
    edges_parser.add_argument("--only", nargs="*", default=None)
    edges_parser.add_argument("--source", default=None)
    args = parser.parse_args()
    counts, edges, sources = read()
    if args.step == "counts":
        report: list[str] = []
        changed = review(args.output, report)
        print("\n".join(report))
        for era, channels in changed.items():
            for channel, quantities in channels.items():
                counts.setdefault(era, {}).setdefault(channel, {}).update(quantities)
        sources["counts"] = args.source or f"statistics review of {args.output}"
        print(f"changed: {json.dumps(changed)}")
    else:
        found = {}
        for line in args.log.read_text().splitlines():
            match = LINE.search(line)
            if match:
                found[match["channel"]] = json.loads(match["json"])
        if not found:
            raise SystemExit(f"{args.log}: no suggested edges (JSON) line")
        if args.only:
            def kept(channel: str, quantity: str) -> bool:
                return any(quantity.startswith(rule.split(":")[-1]) and (":" not in rule or rule.split(":")[0] == channel) for rule in args.only)

            found = {ch: {q: e for q, e in quantities.items() if kept(ch, q)} for ch, quantities in found.items()}
            found = {ch: quantities for ch, quantities in found.items() if quantities}
        edges[args.era] = found
        sources[args.era] = args.source or f"suggest-binning log {args.log.name}"
        print(f"{args.era}: {sum(len(q) for q in found.values())} quantities in {', '.join(found)}")
    write(counts, edges, sources)


if __name__ == "__main__":
    main()
