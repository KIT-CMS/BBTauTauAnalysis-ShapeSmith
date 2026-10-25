"""D6 (nn_limit_spec.md 7.1, NV1; v7 spec D6): the eta-phi occupancy of the tau legs and the jets, data against the
prediction per era.

The occupancy variables of nn_validation.py (occupancy_<object>: the eta x phi cell, |eta| < 2.5 in 20 cells and phi in
24; legs 1 and 2, the two leading jets and the b pair) from the control histograms of a run and region, the prediction
the sum of its backgrounds (with the estimates). Per map: data / prediction and the pull (d - p) / sqrt(p + var_p) after
scaling the prediction to the data (shape only), chi2 over the cells with p >= 5, the cells beyond 3 sigma, and in 2018
data / prediction inside the HEM 15/16 area (eta in [-3.0, -1.3], phi in [-1.57, -0.87], CROWN vetoes events with a jet
there: HEM_ETA, HEM_PHI) against the rest. Plots per era, sample and object.

    python scripts/nn_validation_occupancy.py --sample sr:configs/smrun2_binned_v7b_ff_nn_validation.yaml:et,mt,tt:nominal \\
        --sample btag0:configs/smrun2_binned_v7b_nn_validation_control.yaml:et,mt,tt:btag0_os_pass \\
        --sample dilepton:configs/smrun2_binned_v7b_dilepton_nn_validation.yaml:em,mm:nominal --era ... -o <output>
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from shapesmith import input_gof  # noqa: E402
from shapesmith.config import load_analysis, load_config  # noqa: E402
from shapesmith.histogram import INCLUSIVE, NOMINAL_VARIATION, HistKey, HistogramSet  # noqa: E402

from bbtautau_shapesmith.nn_validation import OCCUPANCY, OCCUPANCY_ETA, OCCUPANCY_N_ETA, OCCUPANCY_N_PHI  # noqa: E402

logger = logging.getLogger("nn_validation_occupancy")
HEM_ETA, HEM_PHI = (-3.0, -1.3), (-1.57, -0.87)
MIN_PREDICTION = 5.0
ETA_EDGES = np.linspace(-OCCUPANCY_ETA, OCCUPANCY_ETA, OCCUPANCY_N_ETA + 1)
PHI_EDGES = np.linspace(-math.pi, math.pi, OCCUPANCY_N_PHI + 1)


def hem_cells() -> np.ndarray:
    """Cells whose centre lies in the HEM 15/16 area, [eta, phi] boolean."""
    eta = (ETA_EDGES[:-1] + ETA_EDGES[1:]) / 2
    phi = (PHI_EDGES[:-1] + PHI_EDGES[1:]) / 2
    return ((eta >= HEM_ETA[0]) & (eta <= HEM_ETA[1]))[:, None] & ((phi >= HEM_PHI[0]) & (phi <= HEM_PHI[1]))[None, :]


def as_map(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(OCCUPANCY_N_ETA, OCCUPANCY_N_PHI)


def compare(data: np.ndarray, prediction: np.ndarray, variance: np.ndarray, era: str) -> dict:
    scale = data.sum() / prediction.sum()
    p, v = prediction * scale, variance * scale**2
    with np.errstate(divide="ignore", invalid="ignore"):
        pull = np.where(p > 0, (data - p) / np.sqrt(p + v), np.nan)
        ratio = np.where(p > 0, data / p, np.nan)
    used = p >= MIN_PREDICTION
    chi2, ndf = float(np.nansum(pull[used] ** 2)), int(used.sum()) - 1
    result = {"scale": float(scale), "chi2": chi2, "ndf": ndf, "p": input_gof.chi2_sf(chi2, ndf) if ndf > 0 else None,
              "cells_beyond_3_sigma": int(np.sum(np.abs(pull[used]) > 3)), "max_abs_pull": float(np.nanmax(np.abs(pull[used]))) if used.any() else None,
              "pull": pull.tolist(), "ratio": ratio.tolist()}
    if era == "2018":
        hem = hem_cells()
        result["hem"] = {"data": float(data[hem].sum()), "prediction": float(p[hem].sum()), "ratio": float(data[hem].sum() / p[hem].sum()) if p[hem].sum() > 0 else None,
                         "ratio_outside": float(data[~hem].sum() / p[~hem].sum())}
    return result


def plot(result: dict, path: Path, title: str, era: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, label, limits in ((axes[0], "ratio", "data / prediction (scaled)", (0.7, 1.3)), (axes[1], "pull", "pull", (-4, 4))):
        mesh = ax.pcolormesh(PHI_EDGES, ETA_EDGES, np.asarray(result[key], dtype=float), cmap="RdBu_r", vmin=limits[0], vmax=limits[1])
        fig.colorbar(mesh, ax=ax, label=label)
        ax.set_xlabel(r"$\varphi$")
        ax.set_ylabel(r"$\eta$")
        if era == "2018":
            ax.add_patch(patches.Rectangle((HEM_PHI[0], HEM_ETA[0]), HEM_PHI[1] - HEM_PHI[0], HEM_ETA[1] - HEM_ETA[0], fill=False, edgecolor="black", linewidth=1.5, linestyle="--"))
    fig.suptitle(f"{title}: chi2/ndf = {result['chi2']:.0f}/{result['ndf']}, p = {result['p']:.2g}" if result["p"] is not None else title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def maps(name: str, config_path: str, channels: list[str], region: str, era: str, output: Path) -> list[dict]:
    config = load_config(config_path, [f"era={era}"])
    analysis = load_analysis(config)
    variables = {f"occupancy_{obj}" for obj in OCCUPANCY}
    hset = HistogramSet.load(config.output_dir / "control_shapes.root", keep=lambda key: key.region == region and key.variable in variables and key.variation == NOMINAL_VARIATION and key.channel in channels)
    results = []
    for channel in channels:
        definition = analysis.channel(channel)
        for variable in sorted(v for v in variables if v in definition.variables):
            data = hset.get(HistKey(channel, INCLUSIVE, definition.data(), region, NOMINAL_VARIATION, variable))
            parts = [h for p in definition.backgrounds() if (h := hset.get(HistKey(channel, INCLUSIVE, p, region, NOMINAL_VARIATION, variable))) is not None]
            if data is None or not parts:
                continue
            prediction = sum(h.values for h in parts)
            variance = sum(h.variances for h in parts)
            if prediction.sum() <= 0:  # the object is absent in the region (the b pair under a b veto)
                logger.info(f"{name} {channel} {era} {variable}: no {variable.removeprefix('occupancy_')} in {region}")
                continue
            result = compare(as_map(data.values), as_map(prediction), as_map(variance), era)
            results.append({"sample": name, "era": era, "channel": channel, "region": region, "variable": variable, **{k: v for k, v in result.items() if k not in ("pull", "ratio")}})
            plot(result, output / "plots" / era / f"{name}_{channel}_{variable}.png", f"{name} {channel} {era}, {variable.removeprefix('occupancy_')}", era)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", action="append", required=True, help="name:run YAML:channels:region")
    parser.add_argument("--era", action="append", required=True)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results = []
    for era in args.era:
        for sample in args.sample:
            name, config, channels, region = sample.split(":")
            results += maps(name, config, channels.split(","), region, era, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "occupancy.json").write_text(json.dumps(results, indent=1))
    lines = ["# D6: eta-phi occupancy, data against the prediction (shape)", "",
             "| sample | channel | era | object | chi2 / ndf | p | cells > 3 sigma | max pull | HEM data/pred (outside) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        hem = f"{r['hem']['ratio']:.3f} ({r['hem']['ratio_outside']:.3f}), data {r['hem']['data']:.0f}" if r.get("hem") and r["hem"]["ratio"] is not None else "-"
        p = f"{r['p']:.2g}" if r["p"] is not None else "-"
        lines.append(f"| {r['sample']} | {r['channel']} | {r['era']} | {r['variable'].removeprefix('occupancy_')} | {r['chi2']:.0f} / {r['ndf']} | {p} | {r['cells_beyond_3_sigma']} | {r['max_abs_pull']:.1f} | {hem} |")
    (args.output / "occupancy.md").write_text("\n".join(lines) + "\n")
    print((args.output / "occupancy.md").read_text())


if __name__ == "__main__":
    main()
