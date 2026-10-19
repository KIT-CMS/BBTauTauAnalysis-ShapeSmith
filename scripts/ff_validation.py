"""Validation of a fake-factor payload of `shapesmith measure` (v7 spec 9.7) on the measurement's own skim:

1. DR closure of the fully corrected fake factor: per leg and process, in the regions where the process is measured
   (QCD and W+jets from data, ttbar from MC), the SR-like yield over the AR-like yield weighted per event with the
   fake factor, the DR->SR correction and the whole compound non-closure correction, in the fake-factor variable and in
   every non-closure variable, per category; chi2 against 1 with the statistical errors of the ratio and its p-value.
   The non-closures are measured in table order, each on top of the earlier ones only; this checks all of them at once.
2. The jet-fake prediction in the application region of each leg (the AR of its fractions): data minus the genuine
   taus and the lepton fakes, weighted with sum_p f_p FF_p DR->SR_p closure_p as the CROWN friend (C15), per b-tag and
   jet category (n_bjets 1 / >= 2 x n_jets 2 / >= 3), with the share of each process; the variants are compared on it.

    python scripts/ff_validation.py -c configs/ff_smrun2_binned_v7b.yaml -s era=2018 [--payload <dir>] [-o <json>]

Writes <payload dir>/validation.json (or -o) and closure plots to <payload dir>/plots/<channel>/closure_*.png.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import correctionlib
import numpy as np
from scipy import stats

from shapesmith.config import load_analysis, load_config
from shapesmith.measurements import MeasureContext
from shapesmith.measurements.fake_factors.measure import ChannelMeasurement, EventSource, Steps

CATEGORIES = {
    "nb1_nj2": "(n_bjets == 1) & (n_jets <= 2)",
    "nb1_nj3p": "(n_bjets == 1) & (n_jets >= 3)",
    "nb2p_nj2": "(n_bjets >= 2) & (n_jets <= 2)",
    "nb2p_nj3p": "(n_bjets >= 2) & (n_jets >= 3)",
}
ALIASES = {"njets": "n_jets", "nbtag": "n_bjets"}


def evaluate(correction, frame) -> np.ndarray:
    """The nominal of a payload correction per event (inputs as float32 -> float64, as the friend)."""
    args = [np.asarray(frame[ALIASES.get(i.name, i.name)], dtype=np.float32).astype(np.float64) for i in correction.inputs[:-1]]
    return correction.evaluate(*args, "nominal")


def fraction(correction, frame, process: str) -> np.ndarray:
    names = [i.name for i in correction.inputs[1:-1]]
    args = [np.asarray(frame[ALIASES.get(n, n)], dtype=np.float32).astype(np.float64) for n in names]
    return correction.evaluate(process, *args, "nominal")


class Payload:
    def __init__(self, directory: Path, channel: str):
        self.ff = correctionlib.CorrectionSet.from_file(str(directory / f"fake_factors_{channel}.json.gz"))
        self.corrections = correctionlib.CorrectionSet.from_file(str(directory / f"FF_corrections_{channel}.json.gz"))
        self.correction_names = set(self.corrections.keys())  # `in` on a CorrectionSet raises for a missing name
        self.compound_names = set(self.corrections.compound.keys())

    def chain(self, name: str) -> list:
        """The fake factor of `name` and its corrections in the order of the friend: DR->SR (if any), the compound."""
        result = [self.ff[f"{name}_fake_factors"]]
        if f"{name}_DR_SR_correction" in self.correction_names:
            result.append(self.corrections[f"{name}_DR_SR_correction"])
        if f"{name}_compound_correction" in self.compound_names:
            result.append(self.corrections.compound[f"{name}_compound_correction"])
        return result

    def weight(self, name: str, frame) -> np.ndarray:
        result = np.ones(len(frame))
        for correction in self.chain(name):
            result = result * evaluate(correction, frame)
        return result


def closure(measurement, source: EventSource, payload: Payload, plots: Path | None) -> list[dict]:
    worker = ChannelMeasurement(measurement, source)
    rows = []
    for leg in measurement.legs[source.channel]:
        for process in leg.processes:
            name = process.name + leg.suffix
            steps = Steps.of(process)
            regions = [(p, steps.ar_like) for p in (steps.target, *steps.subtract)]
            if steps.qcd is not None:
                regions += [(p, steps.qcd.region(steps.ar_like)) for p in (source.data, *steps.qcd.subtract)]
            weights = {(p, r): source.events(p, r).weights * payload.weight(name, source.events(p, r).frame) for p, r in regions}
            for binned in (process.fake_factors, *process.non_closures):
                for category in process.split.categories:
                    ratio = worker.ratio(process.split, binned, category, steps, weights)
                    y, err = ratio.nominal.values, ratio.nominal.errors
                    numerator, denominator = ratio.numerator.values, ratio.denominator.values
                    ok = (denominator > 0) & (numerator > 0) & (err > 0)
                    chi2 = float(np.sum(((y[ok] - 1.0) / err[ok]) ** 2))
                    ndf = int(ok.sum())
                    total = float(np.sum(numerator) / np.sum(denominator)) if np.sum(denominator) else float("nan")
                    row = {
                        "leg": leg.suffix or "leading", "process": name, "variable": binned.variable, "category": category,
                        "split": [process.split.edges[category], process.split.edges[category + 1]], "edges": ratio.nominal.edges.tolist(),
                        "ratio": y.tolist(), "errors": err.tolist(), "chi2": chi2, "ndf": ndf,
                        "p_value": float(stats.chi2.sf(chi2, ndf)) if ndf else None, "total_ratio": total,
                        "max_pull": float(np.max(np.abs((y[ok] - 1.0) / err[ok]))) if ndf else None,
                    }
                    rows.append(row)
                    if plots is not None:
                        _plot(row, plots)
    return rows


def _plot(row: dict, directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    edges = np.asarray(row["edges"])
    centres = 0.5 * (edges[1:] + edges[:-1])
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.axhline(1.0, color="grey", lw=1)
    axis.errorbar(centres, row["ratio"], yerr=row["errors"], xerr=0.5 * np.diff(edges), fmt="o", color="black")
    axis.set_xlabel(row["variable"])
    axis.set_ylabel("SR-like / (AR-like x FF x corrections)")
    p = row["p_value"]
    axis.set_title(f"{row['process']} closure {row['variable']} cat {row['category']}: chi2/ndf {row['chi2']:.1f}/{row['ndf']}" + (f", p {p:.2g}" if p is not None else ""), fontsize=10)
    axis.set_ylim(0.0, 2.0)
    figure.tight_layout()
    figure.savefig(directory / f"closure_{row['process']}_{row['variable']}_category{row['category']}.png")
    plt.close(figure)


def prediction(measurement, source: EventSource, payload: Payload) -> list[dict]:
    """The jet-fake prediction in the application region of every leg, per category, with the process shares."""
    rows = []
    for leg in measurement.legs[source.channel]:
        region = leg.fractions.region
        qcd = leg.processes[0]
        subtract = [p for p in qcd.subtract if p == "EMB" or p.endswith("T") or p.endswith("L")]
        fractions = payload.ff[f"process_fractions{leg.suffix}"]
        processes = [p.name for p in leg.processes]
        key = {"QCD": "QCD", "Wjets": "Wjets", "ttbar": "ttbar"}
        for category, expression in CATEGORIES.items():
            total, shares, raw, nominal_weights = 0.0, dict.fromkeys(processes, 0.0), 0.0, 0.0
            for name, sign in [(source.data, 1.0)] + [(p, -1.0) for p in subtract]:
                events = source.events(name, region)
                frame = events.frame
                mask = frame.eval(expression).to_numpy()
                if not mask.any():
                    continue
                sub = frame[mask]
                w = sign * events.weights[mask]
                nominal_weights += float(np.sum(w))
                for process in processes:
                    f = fraction(fractions, sub, key[process])
                    contribution = w * f * payload.weight(process + leg.suffix, sub)
                    shares[process] += float(np.sum(contribution))
                    raw += float(np.sum(w * f * evaluate(payload.ff[f"{process}{leg.suffix}_fake_factors"], sub)))
            total = sum(shares.values())
            rows.append({
                "leg": leg.suffix or "leading", "category": category, "ar_yield": nominal_weights, "prediction": total,
                "raw_prediction": raw, "mean_ff": total / nominal_weights if nominal_weights else None,
                "shares": {p: (v / total if total else None) for p, v in shares.items()},
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-s", "--set", action="append", default=[])
    parser.add_argument("--channels", default=None)
    parser.add_argument("--payload", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    config = load_config(args.config, args.set)
    analysis = load_analysis(config)
    measurement = analysis.measurement
    directory = args.payload or Path(config.output_dir) / measurement.name / analysis.era
    channels = args.channels.split(",") if args.channels else list(config.channels)
    context = MeasureContext(config, analysis, channels, directory)
    result = {"payload": str(directory), "era": analysis.era, "measurement": measurement.name, "channels": {}}
    for channel in channels:
        source = EventSource(context, channel, measurement.columns(channel))
        payload = Payload(directory, channel)
        plots = None if args.no_plots else directory / "plots" / channel
        result["channels"][channel] = {"closure": closure(measurement, source, payload, plots), "prediction": prediction(measurement, source, payload)}
        print(f"{channel}: done")
    output = args.output or directory / "validation.json"
    output.write_text(json.dumps(result, indent=1))
    print(f"written {output}")


if __name__ == "__main__":
    main()
