"""NV3, the width of the embedding in FastMTT (nn_limit_spec.md 7.1, v7_analyse 11.1 no. 3 and 12.2): EMB against the
simulated genuine di-tau events in m_fastmtt and pt_fastmtt (and m_vis, pt_vis for comparison), et and mt.

From the control histograms of two runs on the same skims: the emb run (EMB in et/mt) and the noemb run (MC genuine
taus: ZTT, TTT, STT, VVT, TTVT, EWKT), in btag0_os_pass (Z -> tautau, the highest purity) and the nominal region. Per
template (the fine bins of the run: 5 GeV in m_fastmtt, 2 GeV in m_vis): the core of an iterative Gaussian (refitted within
+-1.5 sigma, as fastmtt_v7b_summary.md 4), the median, the
interquartile range over the median and the fraction above 120 GeV; the shape compatibility of EMB and MC
(shapesmith.input_gof.shape_compatibility, statistics only) in bins of equal MC prediction; and which template the data
prefer: the saturated GoF p-values of the input-gof runs of both (if given with --gof-emb / --gof-noemb). Plots of the
shapes of EMB and MC in the bins of the shape test per era, channel, region and variable: <output>/plots/.

    python scripts/nn_validation_emb_width.py --emb configs/smrun2_binned_v7b_nn_validation_control.yaml \\
        --noemb-output-dir '/ceph/.../tau_control_nn_noemb/{era}' --era ... -o <output>
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
from shapesmith.config import load_config  # noqa: E402
from shapesmith.histogram import INCLUSIVE, NOMINAL_VARIATION, HistKey, Histogram, HistogramSet  # noqa: E402
from shapesmith.input_gof import GofBinning  # noqa: E402

from bbtautau_shapesmith.processes import EMBEDDED, SPLITS  # noqa: E402

logger = logging.getLogger("nn_validation_emb_width")
GENUINE_MC = tuple(parts["T"] for parts in SPLITS.values())
VARIABLES = ("m_fastmtt", "pt_fastmtt", "m_vis", "pt_vis")
REGIONS = ("btag0_os_pass", "nominal")


def core(h: Histogram, iterations: int = 20) -> tuple[float, float]:
    """mu and sigma of a Gaussian fitted iteratively to the bins within mu +- 1.5 sigma (weighted moments)."""
    centres = (h.edges[:-1] + h.edges[1:]) / 2
    weights = np.clip(h.values, 0.0, None)
    mu = float(centres[np.argmax(weights)])
    sigma = float(np.sqrt(np.average((centres - mu) ** 2, weights=weights)))
    for _ in range(iterations):
        window = np.abs(centres - mu) <= 1.5 * sigma
        if weights[window].sum() <= 0:
            break
        mu = float(np.average(centres[window], weights=weights[window]))
        # the variance of a Gaussian truncated at +-1.5 sigma is 0.6158 of the full one
        sigma = float(np.sqrt(np.average((centres[window] - mu) ** 2, weights=weights[window]) / 0.6158))
    return mu, sigma


def quantiles(h: Histogram, qs=(0.25, 0.5, 0.75)) -> list[float]:
    cumulative = np.cumsum(np.clip(h.values, 0.0, None))
    cumulative /= cumulative[-1]
    return [float(np.interp(q, cumulative, h.edges[1:])) for q in qs]


def describe(h: Histogram, mass: bool) -> dict:
    mu, sigma = core(h)
    low, median, high = quantiles(h)
    result = {"mu": mu, "sigma": sigma, "sigma_over_mu": sigma / mu if mu else None, "median": median, "iqr_over_median": (high - low) / median, "events": h.sum()}
    if mass:
        above = h.edges[:-1] >= 120.0
        result["above_120"] = float(h.values[above].sum() / h.sum())
    return result


def plot(embedded: Histogram, simulated: Histogram, path: Path, title: str, label: str) -> None:
    """The shapes of EMB and MC genuine taus (fraction per unit of the variable) in the bins of the shape test, and their
    ratio; the outer bins (the tails up to the range ends) are cut from the view."""
    edges = np.asarray(embedded.edges)
    widths = np.diff(edges)
    shapes = [h.values / h.sum() / widths for h in (embedded, simulated)]
    errors = [np.sqrt(h.variances) / h.sum() / widths for h in (embedded, simulated)]
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for shape, error, name, color in zip(shapes, errors, ("EMB", "MC genuine tau tau"), ("#e76300", "#3f90da")):
        ax.stairs(shape, edges, label=name, color=color)
        ax.fill_between(edges[:-1], shape - error, shape + error, step="post", color=color, alpha=0.3)
    ax.set_ylabel("fraction per unit")
    ax.set_title(title)
    ax.legend()
    ratio = shapes[0] / shapes[1]
    error = np.hypot(errors[0], errors[1] * ratio) / shapes[1]
    rax.errorbar((edges[:-1] + edges[1:]) / 2, ratio, yerr=error, xerr=widths / 2, fmt="o", markersize=3, color="black")
    rax.axhline(1.0, color="gray")
    rax.set_ylim(0.7, 1.3)
    rax.set_ylabel("EMB / MC")
    rax.set_xlabel(label)
    if len(edges) > 4:
        rax.set_xlim(edges[1], edges[-2])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def summed(hset: HistogramSet, channel: str, region: str, processes, variable: str) -> Histogram | None:
    parts = [h for p in processes if (h := hset.get(HistKey(channel, INCLUSIVE, p, region, NOMINAL_VARIATION, variable))) is not None]
    if not parts:
        return None
    total = parts[0].copy()
    for h in parts[1:]:
        total.add(h)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emb", required=True, help="the run YAML of the emb control run")
    parser.add_argument("--noemb-output-dir", required=True, help="output_dir of the same run with embedding false ({era} filled)")
    parser.add_argument("--era", action="append", required=True)
    parser.add_argument("--gof-emb", type=Path, help="input_gof.json of the emb run in btag0_os_pass")
    parser.add_argument("--gof-noemb", type=Path, help="input_gof.json of the noemb run in btag0_os_pass")
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results = []
    for era in args.era:
        config = load_config(args.emb, [f"era={era}"])
        keep = lambda key: key.region in REGIONS and key.variable in VARIABLES and key.variation == NOMINAL_VARIATION  # noqa: E731
        emb = HistogramSet.load(config.output_dir / "control_shapes.root", keep=keep)
        noemb = HistogramSet.load(Path(args.noemb_output_dir.replace("{era}", era)) / "control_shapes.root", keep=keep)
        for channel in ("et", "mt"):
            for region in REGIONS:
                for variable in VARIABLES:
                    embedded, simulated = summed(emb, channel, region, (EMBEDDED,), variable), summed(noemb, channel, region, GENUINE_MC, variable)
                    if embedded is None or simulated is None:
                        continue
                    edges = input_gof.equal_prediction_edges(simulated.edges, simulated.values, GofBinning(20, 50.0))
                    compatibility = input_gof.shape_compatibility(embedded.rebin(edges), simulated.rebin(edges))
                    plot(embedded.rebin(edges), simulated.rebin(edges), args.output / "plots" / f"{era}_{channel}_{region}_{variable}.png",
                         f"{channel} {era}, {region}: {variable}, shape p = {compatibility['p']:.2g}", f"{variable} / GeV")
                    mass = variable.startswith("m_")
                    results.append({"era": era, "channel": channel, "region": region, "variable": variable, "emb": describe(embedded, mass), "mc": describe(simulated, mass),
                                    "shape_chi2": compatibility["chi2"], "shape_ndf": compatibility["ndf"], "shape_p": compatibility["p"]})
    gof = {}
    for label, path in (("emb", args.gof_emb), ("noemb", args.gof_noemb)):
        if path is not None and path.exists():
            gof[label] = {(e["era"], e["channel"], e["variable"]): e["p"] for e in json.loads(path.read_text())["results"] if e["region"] == "btag0_os_pass"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "emb_width.json").write_text(json.dumps(results, indent=1))
    lines = ["# NV3: the embedding against the simulated genuine di-tau events", "",
             "| era | ch | region | variable | EMB core mu / sigma | MC core mu / sigma | sigma EMB / MC | IQR/median EMB / MC | >120 GeV EMB / MC | shape chi2/ndf (p) | GoF p data: EMB / MC |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        e, m = r["emb"], r["mc"]
        above = f"{e['above_120']:.3f} / {m['above_120']:.3f}" if "above_120" in e else "-"
        fits = "-"
        if r["region"] == "btag0_os_pass" and gof:
            key = (r["era"], r["channel"], r["variable"])
            fits = " / ".join(f"{gof[label][key]:.2g}" if key in gof.get(label, {}) and gof[label][key] is not None else "-" for label in ("emb", "noemb"))
        lines.append(f"| {r['era']} | {r['channel']} | {r['region']} | {r['variable']} | {e['mu']:.1f} / {e['sigma']:.1f} | {m['mu']:.1f} / {m['sigma']:.1f} | {e['sigma'] / m['sigma']:.3f} | "
                     f"{e['iqr_over_median']:.3f} / {m['iqr_over_median']:.3f} | {above} | {r['shape_chi2']:.0f}/{r['shape_ndf']} ({r['shape_p']:.2g}) | {fits} |")
    (args.output / "emb_width.md").write_text("\n".join(lines) + "\n")
    print((args.output / "emb_width.md").read_text())


if __name__ == "__main__":
    main()
