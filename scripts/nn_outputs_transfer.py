"""NV4, FF closure per NN category (nn_limit_spec.md 7.1; nn_outputs.py): are the NN outputs of the simulated jet
fakes distributed alike in the application region (AR) times the fake factor and in the signal region (SR), within
the fake-factor uncertainties? The same test as the MC jet fakes of NV1 (scripts/nn_validation_transfer.py), on the
outputs of the model instead of its inputs.

The jet -> tau_h fakes of the simulation (the J parts of DY, ttbar, single t, VV, ttV, EWK and W; no simulated QCD, so
in tt the non-QCD fakes only) in the nominal region against the same processes in the anti-isolated region times the
fake factor of ff_v7b (run smrun2_binned_v7b_nn_m0k4_nv4_control.yaml, switch ff_transfer, region anti_iso), per
channel, era and output:
- NN_cat_<category>: the score in each NN category (the HH categories from their MC-only copies NNmc_cat_, every
  score: no data enters);
- NN_P_<class>: the score of each class over every event (from NNmc_P_);
- NN_<input>_<category>: n_jets and mass_2 in each category (NV1: mass_2 fails this test inclusively);
- NN_categories: the share of each category (the migration between the categories).
Test (shapesmith.input_gof.shape_compatibility): the unit-sum shapes of SR and AR x FF, chi2 with the statistical
covariance of both and every fake-factor shift of AR x FF (CMS_ff_*: the friend shifts and the decay-mode
non-closure) as d d^T, n - 1 degrees of freedom; the shape only, as NV1 (the fake factor of the data need not close
in normalisation on simulation without QCD). Bins: at most 10 of equal SR prediction, merged until each holds >= 25
effective events in both regions. Per test the shape ratio SR / (AR x FF) and the fake-factor band per bin
(transfer.json), the size a non-closure shape nuisance would need. The failure rule of NV1 over these tests (or over
--n-tests, the N of the whole NV4 family).

    python scripts/nn_outputs_transfer.py -c configs/smrun2_binned_v7b_nn_m0k4_nv4_control.yaml --era ... -o <output>
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
from bbtautau_shapesmith.nn_outputs import INPUTS  # noqa: E402
from bbtautau_shapesmith.processes import JET_FAKE  # noqa: E402
from bbtautau_shapesmith.variables import NN_SIGNAL_CLASS  # noqa: E402

logger = logging.getLogger("nn_outputs_transfer")
SR, AR = "nominal", "anti_iso"
REGION = "mc_jet_fakes"
BINNING = GofBinning(10, 0.0)
MIN_EFFECTIVE = 25.0
CATEGORIES = "NN_categories"


def summed(hset: HistogramSet, channel: str, region: str, processes, variable: str, variation: str = NOMINAL_VARIATION) -> Histogram | None:
    """The sum of the processes; a process without the variation counts with its nominal histogram."""
    total = None
    for process in processes:
        h = hset.get(HistKey(channel, INCLUSIVE, process, region, variation, variable)) or hset.get(HistKey(channel, INCLUSIVE, process, region, NOMINAL_VARIATION, variable))
        if h is not None:
            total = h.copy() if total is None else total.add(h)
    return total


def outputs(channel) -> dict[str, str]:
    """Test name -> the control variable it reads: the MC-only copy where the variable of the data hides events (the
    HH categories, the class scores)."""
    names = {}
    categories = [name[len("NN_cat_"):] for name in channel.variables if name.startswith("NN_cat_")]
    for category in categories:
        source = "NNmc" if category.startswith(NN_SIGNAL_CLASS) else "NN"
        names[f"NN_cat_{category}"] = f"{source}_cat_{category}"
        for name in INPUTS:
            names[f"NN_{name}_{category}"] = f"{source}_{name}_{category}"
    for name in channel.variables:
        if name.startswith("NNmc_P_"):
            names[f"NN_P_{name[len('NNmc_P_'):]}"] = name
    return names


def compare(sr: Histogram, ar: Histogram, shifts: dict[str, tuple[Histogram, Histogram]]) -> dict:
    """shape_compatibility with and without the shifts, the shapes and the fake-factor band per bin."""
    with_ff = input_gof.shape_compatibility(sr, ar, shifts)
    stat = input_gof.shape_compatibility(sr, ar)
    shape_sr, shape_ar = sr.values / sr.sum(), ar.values / ar.sum()
    band = np.sqrt(sum(((u.values / u.sum() - d.values / d.sum()) / 2) ** 2 for u, d in shifts.values())) if shifts else np.zeros(len(shape_ar))
    return {"p": with_ff["p"], "method": "chi2", "chi2": with_ff["chi2"], "ndf": with_ff["ndf"], "p_stat": stat["p"], "shifts": with_ff["shifts"],
            "ratio": with_ff["ratio"], "sr_shape": shape_sr.tolist(), "ar_shape": shape_ar.tolist(), "ff_band": (band / np.where(shape_ar > 0, shape_ar, np.nan)).tolist(),
            "sr_stat": (np.sqrt(sr.variances) / sr.sum() / np.where(shape_ar > 0, shape_ar, np.nan)).tolist(), "sr": sr.sum(), "ar_ff": ar.sum()}


def category_yields(hset: HistogramSet, channel: str, region: str, processes, variables: list[str], variation: str = NOMINAL_VARIATION) -> Histogram | None:
    """The yield (and its variance) of every category as one bin each."""
    totals = [summed(hset, channel, region, processes, variable, variation) for variable in variables]
    if any(total is None for total in totals):
        return None
    return Histogram(np.arange(len(totals) + 1, dtype=float), [t.sum() for t in totals], [t.variances.sum() for t in totals])


def tests_of(config_path: str, overrides: list[str], era: str, channels: list[str], output: Path) -> list[dict]:
    config = load_config(config_path, [f"era={era}", *overrides])
    analysis = load_analysis(config)
    hset = HistogramSet.load(Path(config.output_dir) / "control_shapes.root", keep=lambda key: key.region in (SR, AR) and key.process in JET_FAKE)
    results = []
    for channel_name in channels:
        channel = analysis.channel(channel_name)
        processes = [p.name for p in channel.processes if p.name in JET_FAKE]
        shift_names = sorted({k.variation[:-2] for k in hset.select(channel=channel_name, region=AR) if k.variation.startswith("CMS_ff_") and k.variation.endswith("Up")})
        tests = outputs(channel)
        for name, variable in tests.items():
            sr, ar = summed(hset, channel_name, SR, processes, variable), summed(hset, channel_name, AR, processes, variable)
            if sr is None or ar is None or sr.sum() <= 0 or ar.sum() <= 0:
                logger.warning(f"{era} {channel_name} {name}: no MC jet fakes")
                continue
            edges = input_gof.equal_prediction_edges(sr.edges, sr.values, BINNING, input_gof.is_integer_binning(sr.edges))
            edges = input_gof.merge_low_statistics(edges, [sr.rebin(edges), ar.rebin(edges)], MIN_EFFECTIVE)
            if len(edges) < 3:
                logger.info(f"{era} {channel_name} {name}: one bin, no test")
                continue
            shifts = {s: tuple(summed(hset, channel_name, AR, processes, variable, f"{s}{d}").rebin(edges) for d in ("Up", "Down")) for s in shift_names}
            result = compare(sr.rebin(edges), ar.rebin(edges), shifts)
            results.append({"era": era, "channel": channel_name, "region": REGION, "variable": name, "source": variable, "edges": edges, **result})
            if result["p"] is not None and result["p"] < 0.01:
                plot(result, output / "plots" / f"{era}_{channel_name}_{name}.png", f"{channel_name} {era}: {name}, p = {result['p']:.2g}")
        scores = [variable for name, variable in tests.items() if name.startswith("NN_cat_")]
        sr, ar = category_yields(hset, channel_name, SR, processes, scores), category_yields(hset, channel_name, AR, processes, scores)
        if sr is not None and ar is not None:
            shifts = {s: tuple(category_yields(hset, channel_name, AR, processes, scores, f"{s}{d}") for d in ("Up", "Down")) for s in shift_names}
            result = compare(sr, ar, shifts)
            results.append({"era": era, "channel": channel_name, "region": REGION, "variable": CATEGORIES, "source": scores, "edges": sr.edges.tolist(), **result})
    return results


def plot(result: dict, path: Path, title: str) -> None:
    """The unit-sum shapes of the MC jet fakes in SR and AR x FF (bin index) and their ratio with the FF band."""
    shape_sr, shape_ar = np.array(result["sr_shape"]), np.array(result["ar_shape"])
    band, stat = np.array(result["ff_band"], dtype=float), np.array(result["sr_stat"], dtype=float)
    x = np.arange(len(shape_sr))
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax.step(x, shape_sr, where="mid", label="SR (MC jet fakes)", color="black")
    ax.step(x, shape_ar, where="mid", label="AR x FF", color="#3f90da")
    ax.fill_between(x, shape_ar * (1 - band), shape_ar * (1 + band), step="mid", color="#3f90da", alpha=0.3, label="FF uncertainty")
    ax.legend()
    ax.set_title(title)
    rax.errorbar(x, np.array(result["ratio"], dtype=float), yerr=stat, fmt="o", color="black")
    rax.fill_between(x, 1 - band, 1 + band, step="mid", color="#3f90da", alpha=0.3)
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
    parser.add_argument("-s", "--set", dest="overrides", action="append", default=[], help="override of the run YAML (repeatable)")
    parser.add_argument("--era", action="append", required=True)
    parser.add_argument("--channels", default=",".join(TAU_CHANNELS))
    parser.add_argument("--n-tests", type=int, default=None, help="N of the failure rule (default: these tests)")
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results = []
    for era in args.era:
        results += tests_of(args.config, args.overrides, era, args.channels.split(","), args.output)
    rule = input_gof.failure_rule(results, n_tests=args.n_tests)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "transfer.json").write_text(json.dumps({"results": results, "rule": {k: v for k, v in rule.items() if k != "entries"}}, indent=1))
    lines = input_gof.summary_lines({"results": results, "calibration": {}}, rule, "NV4, FF closure per NN category: MC jet fakes, AR x FF against SR (shape, with the FF uncertainties)")
    stat_failing = sum(1 for e in results if e["p_stat"] is not None and e["p_stat"] < rule["threshold"])
    lines += ["", f"Statistics only (no FF uncertainty): {stat_failing} of {len(results)} tests below alpha / N."]
    (args.output / "transfer.md").write_text("\n".join(line.replace("ᶠ", "") for line in lines if not line.startswith("p-values:")) + "\n")
    print((args.output / "transfer.md").read_text())


if __name__ == "__main__":
    main()
