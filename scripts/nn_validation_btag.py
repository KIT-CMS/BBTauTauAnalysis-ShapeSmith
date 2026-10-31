"""NV2 (nn_limit_spec.md 7.1, E15, N13): are the b-tag scores of the NN inputs modelled within each working-point bin?

The scores bpair_btag_value_{1,2} and jtag_value_{1,2} (UParTAK4B) per era in the WP bins < M, M-T and >= T of the era
(nn_validation.BTAG_WP, from the BTV payload CROWN pins), in
- em_tt btag1 and btag2 (the b-rich ttbar control, regions btag1/btag2 of smrun2_binned_v7b_dilepton_nn_validation.yaml),
- mm and ee with >= 2 jets (light jets; its nominal region),
- the signal region of et, mt and tt (smrun2_binned_v7b_ff_nn_validation.yaml, EMB + jetFakes + MC).
Test (shapesmith.input_gof.flatness): in each WP bin the prediction scaled to the data of the bin (the normalisation is
the SF's, not tested), its fine bins merged to at most 10 bins of equal prediction (>= 20 events), chi2 of data against
prediction with the statistics of both (n - 1 degrees of freedom), and the slope of data / prediction across the bin.
Failure rule as NV1 (shapesmith.input_gof.failure_rule): p < 0.05 / N over the N tests, or p < 0.05 in >= 2 eras of one
score, region and WP bin. Verdict (N13): continuous inputs if no test fails, else WP-binned inputs everywhere. As a
diagnostic the same chi2 with the b-tag SF variations of the run as shape uncertainties (p_shifts).

    python scripts/nn_validation_btag.py --dilepton configs/smrun2_binned_v7b_dilepton_nn_validation.yaml \\
        --sr configs/smrun2_binned_v7b_ff_nn_validation.yaml --era 2016preVFP ... -o <output>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from shapesmith import input_gof  # noqa: E402
from shapesmith.config import load_analysis, load_config  # noqa: E402
from shapesmith.histogram import INCLUSIVE, NOMINAL_VARIATION, HistKey, Histogram, HistogramSet  # noqa: E402
from shapesmith.input_gof import GofBinning  # noqa: E402

from bbtautau_shapesmith.nn_validation import BTAG_SCORES, BTAG_WP  # noqa: E402

logger = logging.getLogger("nn_validation_btag")
SAMPLES = {"dilepton": (("em", "btag1"), ("em", "btag2"), ("mm", "nominal"), ("ee", "nominal")), "sr": (("et", "nominal"), ("mt", "nominal"), ("tt", "nominal"))}
WP_BINS = ("<M", "M-T", ">=T")
BINNING = GofBinning(10, 20.0)


def summed(hset: HistogramSet, channel: str, region: str, processes, variable: str, variation: str = NOMINAL_VARIATION) -> Histogram | None:
    total = None
    for process in processes:
        h = hset.get(HistKey(channel, INCLUSIVE, process, region, variation, variable)) or hset.get(HistKey(channel, INCLUSIVE, process, region, NOMINAL_VARIATION, variable))
        if h is not None:
            total = h.copy() if total is None else total.add(h)
    return total


def btag_shifts(hset: HistogramSet, channel: str, region: str, processes, variable: str) -> dict[str, tuple[Histogram, Histogram]]:
    names = sorted({k.variation[:-2] for k in hset.select(channel=channel, region=region, variable=variable) if k.variation.startswith("CMS_btag") and k.variation.endswith("Up")})
    return {name: (summed(hset, channel, region, processes, variable, f"{name}Up"), summed(hset, channel, region, processes, variable, f"{name}Down")) for name in names}


def tests_of(config_path: str, era: str, kind: str) -> list[dict]:
    config = load_config(config_path, [f"era={era}"])
    analysis = load_analysis(config)
    channels = {channel for channel, _ in SAMPLES[kind]}
    hset = HistogramSet.load(config.output_dir / "control_shapes.root", keep=lambda key: key.channel in channels and key.variable in BTAG_SCORES)
    medium, tight = BTAG_WP[era]
    ranges = [(0.0, medium), (medium, tight), (tight, 1.0)]
    results = []
    for channel, region in SAMPLES[kind]:
        definition = analysis.channel(channel)
        processes = definition.backgrounds()
        for variable in BTAG_SCORES:
            data = hset.get(HistKey(channel, INCLUSIVE, definition.data(), region, NOMINAL_VARIATION, variable))
            prediction = summed(hset, channel, region, processes, variable)
            if data is None or prediction is None:
                logger.warning(f"{era} {channel} {region} {variable}: no histograms")
                continue
            shifts = btag_shifts(hset, channel, region, processes, variable)
            for wp, entry in zip(WP_BINS, input_gof.flatness(data, prediction, ranges, BINNING, shifts)):
                results.append({"era": era, "channel": f"{channel}_{region}", "region": wp, "variable": variable, **entry, "shift_names": sorted(shifts)})
    return results


def plot(results: list[dict], output: Path) -> None:
    """Per sample and score: data / prediction (each WP bin normalised) against the prediction quantile, one panel per
    WP bin, the eras overlaid."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for entry in results:
        groups.setdefault((entry["channel"], entry["variable"]), []).append(entry)
    colours = {"2016preVFP": "#3f90da", "2016postVFP": "#ffa90e", "2017": "#bd1f01", "2018": "#832db6"}
    for (sample, variable), entries in groups.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, wp in zip(axes, WP_BINS):
            for entry in (e for e in entries if e["region"] == wp and e.get("bins", 0) > 1):
                ax.errorbar(entry["x"], entry["ratio"], yerr=entry["ratio_error"], fmt="o-", markersize=3, color=colours[entry["era"]],
                            label=f"{entry['era']}: p = {entry['p']:.2g}, slope {entry['slope']:+.3f}" if entry["slope"] is not None else entry["era"])
            ax.axhline(1.0, color="gray", linewidth=1)
            ax.set_title(f"{wp}")
            ax.set_xlabel("prediction quantile in the WP bin")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("data / prediction (normalised)")
        axes[0].set_ylim(0.7, 1.3)
        fig.suptitle(f"{variable}, {sample}")
        fig.tight_layout()
        path = output / "plots" / f"{sample}_{variable}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)


def lines(results: list[dict], rule: dict, eras: list[str]) -> list[str]:
    out = ["# NV2: b-tag scores within the working-point bins", "",
           f"{rule['tests']} tests (score x sample x era x WP bin), alpha / N = {rule['threshold']:.3g}; p < 0.05: {rule['below_alpha']} (expected {rule['expected_below_alpha']:.1f}); "
           f"KS of the p-values {rule['ks_p_value']:.3g}", ""]
    failed = [entry for entry in rule["entries"] if entry["bonferroni"]]
    by_era = {era: [e for e in rule["entries"] if e["era"] == era] for era in eras}
    out += ["| era | tests | p < alpha / N | p < 0.05 | verdict of the era |", "|---|---|---|---|---|"]
    for era, entries in by_era.items():
        bad = sum(e["bonferroni"] for e in entries)
        out.append(f"| {era} | {len(entries)} | {bad} | {sum(e['below_alpha'] for e in entries)} | {'WP-binned' if bad else 'continuous'} |")
    two_eras = [f"{name}: {'; '.join(record['reasons'])}" for name, record in rule["variables"].items() if record["failed"]]
    out += ["", f"**Global verdict (N13): {'WP-binned b-tag inputs everywhere' if two_eras else 'continuous b-tag inputs'}**", ""]
    out += [f"- {line}" for line in two_eras] + [""]
    out += ["| score | sample | WP | " + " | ".join(eras) + " |", "|---|---|---|" + "---|" * len(eras)]
    table = {(e["variable"], e["channel"], e["region"], e["era"]): e for e in results}
    for variable in BTAG_SCORES:
        for sample in dict.fromkeys(e["channel"] for e in results):
            for wp in WP_BINS:
                cells = []
                for era in eras:
                    e = table.get((variable, sample, wp, era))
                    if e is None or e.get("p") is None:
                        cells.append("-")
                        continue
                    mark = " **!!**" if e["p"] < rule["threshold"] else (" *" if e["p"] < 0.05 else "")
                    shift = f", syst {e['p_shifts']:.2g}" if e.get("p_shifts") is not None else ""
                    cells.append(f"{e['p']:.2g}{mark} ({e['slope']:+.3f}{shift}; {e['data']:.0f} ev.)" if e["slope"] is not None else f"{e['p']:.2g}{mark}")
                out.append(f"| {variable} | {sample} | {wp} | " + " | ".join(cells) + " |")
    out += ["", "Cells: p-value of the flatness chi2 (statistics only), * p < 0.05, **!!** p < alpha / N; in brackets the slope of data/prediction "
            "across the WP bin (per unit of the prediction quantile), the p-value with the b-tag SF variations as shape uncertainties, the data events.",
            f"Failing tests: {len(failed)}"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dilepton", required=True)
    parser.add_argument("--sr", help="the SR run YAML (without it: the dilepton samples only)")
    parser.add_argument("--era", action="append", required=True)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results = []
    for era in args.era:
        results += tests_of(args.dilepton, era, "dilepton") + (tests_of(args.sr, era, "sr") if args.sr else [])
    entries = [{**e, "variable": f"{e['variable']}"} for e in results if e.get("p") is not None]
    rule = input_gof.failure_rule([{**e, "variable": f"{e['variable']} {e['channel']} {e['region']}"} for e in entries])
    # the per-test flags of the rule on the entries as tested
    for entry, flagged in zip(entries, rule["entries"]):
        entry.update(bonferroni=flagged["bonferroni"], below_alpha=flagged["below_alpha"])
    rule_view = {**rule, "entries": entries}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "nv2.json").write_text(json.dumps({"results": results, "rule": {k: v for k, v in rule.items() if k != "entries"}}, indent=1))
    (args.output / "nv2.md").write_text("\n".join(lines(results, rule_view, args.era)) + "\n")
    plot(results, args.output)
    print((args.output / "nv2.md").read_text())


if __name__ == "__main__":
    main()
