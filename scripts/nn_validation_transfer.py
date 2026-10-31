"""NV1, MC jet fakes (nn_limit_spec.md 7.1): is every input distributed alike in the application region (AR) and the
signal region (SR), within the fake-factor uncertainty?

The jet -> tau_h fakes of the simulation (the J parts of DY, ttbar, single t, VV, ttV, EWK and W; there is no simulated
QCD, so in tt this covers the non-QCD fakes only) in the nominal region (SR) against the same processes in the
anti-isolated region times the fake factor of the friend ff_v7b (switch ff_transfer of
smrun2_binned_v7b_nn_validation_control.yaml, region anti_iso), per input, channel and era. Test
(shapesmith.input_gof.shape_compatibility): the unit-sum shapes of SR and AR x FF, chi2 with the statistical
covariance of both and the fake-factor friend shifts of AR x FF (CMS_ff_*: statistics, systematic bands, MC
subtraction, corrections, fractions) as d d^T, n - 1 degrees of freedom; the shape only (the fake factor of data need
not close in normalisation on MC). Bins: at most 10 of equal SR prediction, merged until each holds >= 25 effective
events in both regions. The failure rule of NV1 over all these tests; the stat-only p-value as a diagnostic.

    python scripts/nn_validation_transfer.py -c configs/smrun2_binned_v7b_nn_validation_control.yaml --era ... -o <output>
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

from bbtautau_shapesmith.constants import TAU_CHANNELS  # noqa: E402
from bbtautau_shapesmith.nn_validation import inputs  # noqa: E402
from bbtautau_shapesmith.processes import JET_FAKE  # noqa: E402

logger = logging.getLogger("nn_validation_transfer")
SR, AR = "nominal", "anti_iso"
BINNING = GofBinning(10, 0.0)
MIN_EFFECTIVE = 25.0


def summed(hset: HistogramSet, channel: str, region: str, processes, variable: str, variation: str = NOMINAL_VARIATION) -> Histogram | None:
    total = None
    for process in processes:
        h = hset.get(HistKey(channel, INCLUSIVE, process, region, variation, variable)) or hset.get(HistKey(channel, INCLUSIVE, process, region, NOMINAL_VARIATION, variable))
        if h is not None:
            total = h.copy() if total is None else total.add(h)
    return total


def tests_of(config_path: str, era: str, channels: list[str], output: Path) -> list[dict]:
    config = load_config(config_path, [f"era={era}"])
    analysis = load_analysis(config)
    hset = HistogramSet.load(config.output_dir / "control_shapes.root", keep=lambda key: key.region in (SR, AR) and key.process in JET_FAKE)
    results = []
    for channel in channels:
        processes = [p.name for p in analysis.channel(channel).processes if p.name in JET_FAKE]
        shift_names = sorted({k.variation[:-2] for k in hset.select(channel=channel, region=AR) if k.variation.startswith("CMS_ff_") and k.variation.endswith("Up")})
        for variable in inputs(channel):
            sr, ar = summed(hset, channel, SR, processes, variable), summed(hset, channel, AR, processes, variable)
            if sr is None or ar is None or sr.sum() <= 0 or ar.sum() <= 0:
                logger.warning(f"{era} {channel} {variable}: no MC jet fakes")
                continue
            edges = input_gof.equal_prediction_edges(sr.edges, sr.values, BINNING, input_gof.is_integer_binning(sr.edges))
            edges = input_gof.merge_low_statistics(edges, [sr.rebin(edges), ar.rebin(edges)], MIN_EFFECTIVE)
            shifts = {name: tuple(summed(hset, channel, AR, processes, variable, f"{name}{d}").rebin(edges) for d in ("Up", "Down")) for name in shift_names}
            with_ff = input_gof.shape_compatibility(sr.rebin(edges), ar.rebin(edges), shifts)
            stat = input_gof.shape_compatibility(sr.rebin(edges), ar.rebin(edges))
            results.append({"era": era, "channel": channel, "region": "mc_jet_fakes", "variable": variable, "p": with_ff["p"], "method": "chi2", "chi2": with_ff["chi2"], "ndf": with_ff["ndf"],
                            "p_stat": stat["p"], "edges": edges, "sr": sr.sum(), "ar_ff": ar.sum(), "shifts": with_ff["shifts"], "ratio": with_ff["ratio"]})
            if with_ff["p"] is not None and with_ff["p"] < 0.01:
                plot(sr.rebin(edges), ar.rebin(edges), shifts, output / "plots" / f"{era}_{channel}_{variable}.png", f"{channel} {era}: {variable}, p = {with_ff['p']:.2g}")
    return results


def plot(sr: Histogram, ar: Histogram, shifts: dict, path: Path, title: str) -> None:
    """The unit-sum shapes of the MC jet fakes in SR and AR x FF (bin index) and their ratio with the FF band."""
    shape_sr, shape_ar = sr.values / sr.sum(), ar.values / ar.sum()
    band = np.sqrt(sum(((u.values / u.sum() - d.values / d.sum()) / 2) ** 2 for u, d in shifts.values())) if shifts else 0 * shape_ar
    x = np.arange(len(shape_sr))
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax.step(x, shape_sr, where="mid", label="SR (MC jet fakes)", color="black")
    ax.step(x, shape_ar, where="mid", label="AR x FF", color="#3f90da")
    ax.fill_between(x, shape_ar - band, shape_ar + band, step="mid", color="#3f90da", alpha=0.3, label="FF uncertainty")
    ax.legend()
    ax.set_title(title)
    ratio = shape_sr / np.where(shape_ar > 0, shape_ar, np.nan)
    error = np.sqrt(sr.variances) / sr.sum() / np.where(shape_ar > 0, shape_ar, np.nan)
    rax.errorbar(x, ratio, yerr=error, fmt="o", color="black")
    rax.fill_between(x, 1 - band / np.where(shape_ar > 0, shape_ar, np.nan), 1 + band / np.where(shape_ar > 0, shape_ar, np.nan), step="mid", color="#3f90da", alpha=0.3)
    rax.axhline(1.0, color="gray")
    rax.set_ylim(0.5, 1.5)
    rax.set_xlabel("bin (equal SR prediction)")
    rax.set_ylabel("SR / (AR x FF)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--era", action="append", required=True)
    parser.add_argument("--channels", default=",".join(TAU_CHANNELS))
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results = []
    for era in args.era:
        results += tests_of(args.config, era, args.channels.split(","), args.output)
    rule = input_gof.failure_rule(results)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "transfer.json").write_text(json.dumps({"results": results, "rule": {k: v for k, v in rule.items() if k != "entries"}}, indent=1))
    summary = {"results": results, "calibration": {}}
    lines = input_gof.summary_lines(summary, rule, "NV1, MC jet fakes: AR x FF against SR (shape, with the FF uncertainties)")
    stat_failing = sum(1 for e in results if e["p_stat"] is not None and e["p_stat"] < rule["threshold"])
    lines += ["", f"Statistics only (no FF uncertainty): {stat_failing} of {len(results)} tests below alpha / N."]
    (args.output / "transfer.md").write_text("\n".join(line.replace("ᶠ", "") for line in lines if not line.startswith("p-values:")) + "\n")
    print((args.output / "transfer.md").read_text())


if __name__ == "__main__":
    main()
