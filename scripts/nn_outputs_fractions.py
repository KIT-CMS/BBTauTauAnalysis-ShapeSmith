"""NV4b, the fake-factor fractions per NN category (nn_limit_spec.md 7.1; nn_outputs.py): the fraction of ttbar jet
fakes the fake factor applies in each NN category and score bin (its payload is binned in (mt_1, n_jets) for et/mt
and (m_vis, n_jets) for tt, not by NN category) against the fraction in the anti-isolated region (AR) of that bin.

Per era, channel, NN category and card bin (the edges of the datacards, nn_binning.json; every score of the HH
category: the AR holds no event of the signal region):
- truth, from the mc-mode run (smrun2_binned_v7b_nn_m0k4_nv4_control.yaml), region abcd_anti_iso (the AR without the
  fake factor; variables NN_cat_<category>, the HH categories NN_ar_<category>): in the definition of the payload
  (shapesmith FF measurement) QCD = data - every simulated process (negative 0), f_ttbar = TTJ / (QCD + TTJ), with its
  statistical uncertainty; and the composition of the jet fakes data - genuine - lepton fakes: ttbar-J (TTJ), W
  (the W+jets fakes), other simulated J (ZJ, STJ, VVJ, TTVJ, EWKJ) and QCD (data - every simulated process);
- payload: the fraction the friend applies, the average of the payload's f_ttbar over the AR jet fakes of the bin,
  (data - genuine - lepton fakes) in the region anti_iso_ttbar_fraction (weighted by f_ttbar) over the same unweighted,
  and its frac_ttbar_J_up shift (anti_iso_ttbar_fraction_up): delta_f;
- jetFakes template, from the fake-factor run of the limit (its shapes.root): the nominal T per bin, the shift of the
  payload's ttbar fractions T(frac_ttbar_J Up) - T (in tt both legs), and the fake-factor band sigma_FF (every
  CMS_ff_* shift of jetFakes, (Up - Down) / 2 in quadrature) and its statistics.
The fake factor is linear in the fractions (f_QCD = 1 - f_ttbar), so replacing the payload's fraction by the truth
of the bin moves the template by dT = (T(frac Up) - T) x (f_truth - f_payload) / delta_f, exact for a delta_f constant
over the payload cells of the bin. Verdict per bin: |dT| <= sigma_FF (within the fake-factor uncertainty); and the
significance of f_truth - f_payload in its statistical uncertainty.

    python scripts/nn_outputs_fractions.py -c configs/smrun2_binned_v7b_nn_m0k4_nv4_control.yaml --era ... \\
        --limit-run <output_dir of the fake-factor run, {era}> -o <output>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from shapesmith.config import load_analysis, load_config
from shapesmith.histogram import INCLUSIVE, NOMINAL_VARIATION, HistKey, Histogram, HistogramSet

from bbtautau_shapesmith.constants import TAU_CHANNELS
from bbtautau_shapesmith.nn_outputs import FRACTION_REGION
from bbtautau_shapesmith.processes import EMBEDDED, GENUINE, JET_FAKE, LEPTON_FAKE, SPLITS
from bbtautau_shapesmith.variables import NN_SIGNAL_CLASS

logger = logging.getLogger("nn_outputs_fractions")
AR = "abcd_anti_iso"
TTBAR_J, W = SPLITS["TT"]["J"], "W"
# the payload's ttbar-fraction shifts in the names of the card nuisances (tt: the leading and the subleading tau)
FRACTION_SHIFTS = ("CMS_ff_process_fractionsfrac_ttbar_J_", "CMS_ff_process_fractions_subleadingfrac_ttbar_J_")


def _sum(hset: HistogramSet, channel: str, region: str, processes, variable: str, edges) -> tuple[np.ndarray, np.ndarray]:
    values, variances = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    for process in processes:
        h = hset.get(HistKey(channel, INCLUSIVE, process, region, NOMINAL_VARIATION, variable))
        if h is not None:
            rebinned = h.rebin(edges)
            values, variances = values + rebinned.values, variances + rebinned.variances
    return values, variances


def truth(hset: HistogramSet, analysis, channel: str, variable: str, edges) -> dict:
    """The AR composition per bin (module docstring)."""
    processes = [p.name for p in analysis.channel(channel).processes if p.role == "background"]
    genuine = [p for p in processes if p in GENUINE]
    lepton = [p for p in processes if p in LEPTON_FAKE]
    jets = [p for p in processes if p in JET_FAKE]
    others = [p for p in jets if p not in (TTBAR_J, W)]
    data, data_var = _sum(hset, channel, AR, ["data"], variable, edges)
    parts = {name: _sum(hset, channel, AR, members, variable, edges) for name, members in
             (("genuine", genuine), ("lepton", lepton), ("ttbar_J", [TTBAR_J]), ("W", [W]), ("other_J", others))}
    simulated = sum(v for v, _ in parts.values())
    simulated_var = sum(v for _, v in parts.values())
    qcd = data - simulated
    qcd_payload = np.clip(qcd, 0.0, None)
    ttbar, ttbar_var = parts["ttbar_J"]
    denominator = qcd_payload + ttbar
    with np.errstate(divide="ignore", invalid="ignore"):
        f_ttbar = np.where(denominator > 0, ttbar / denominator, np.nan)
        # f = t / (Q + t) = t / (D - S), S every simulated process but TTJ: df/dt = 1 / den, df/dD = -t / den^2 = -df/dS
        others_var = simulated_var - ttbar_var
        f_error = np.sqrt((ttbar / denominator**2) ** 2 * (data_var + others_var) + (1 / denominator) ** 2 * ttbar_var)
        fakes = data - parts["genuine"][0] - parts["lepton"][0]
        shares = {name: (parts[name][0] / fakes).tolist() for name in ("ttbar_J", "W", "other_J")}
        shares["QCD"] = (qcd / fakes).tolist()
    return {"data": data.tolist(), "fakes": fakes.tolist(), "f_ttbar": f_ttbar.tolist(), "f_ttbar_error": f_error.tolist(), "shares": shares,
            "yields": {name: value.tolist() for name, (value, _) in parts.items()}}


def payload_fraction(hset: HistogramSet, analysis, channel: str, variable: str, edges) -> dict:
    """The payload's f_ttbar and its frac_ttbar_J_up shift averaged over the AR jet fakes of each bin."""
    processes = [p.name for p in analysis.channel(channel).processes if p.role == "background" and (p.name in GENUINE or p.name in LEPTON_FAKE)]

    def fakes(region):
        data, _ = _sum(hset, channel, region, ["data"], variable, edges)
        subtracted, _ = _sum(hset, channel, region, processes, variable, edges)
        return data - subtracted

    unweighted = fakes(AR)
    with np.errstate(divide="ignore", invalid="ignore"):
        nominal = fakes(FRACTION_REGION) / unweighted
        up = fakes(f"{FRACTION_REGION}_up") / unweighted
    return {"f_ttbar": nominal.tolist(), "f_ttbar_up": up.tolist(), "delta_f": (up - nominal).tolist()}


def template(limit: HistogramSet, channel: str, category: str, edges) -> dict:
    """jetFakes of the fit category per card bin: nominal, its statistics, the shift of the payload's ttbar fractions
    and the band of every fake-factor shift."""
    keys = [k for k in limit.select(channel=channel, category=category, process="jetFakes", region="nominal")]
    nominal = limit[HistKey(channel, category, "jetFakes", "nominal", NOMINAL_VARIATION, "NN_score")].rebin(edges)
    shifts = sorted({k.variation[:-2] for k in keys if k.variation.startswith("CMS_ff_") and k.variation.endswith("Up")})
    band = np.zeros(len(edges) - 1)
    fraction_shift = np.zeros(len(edges) - 1)
    for name in shifts:
        up = limit[HistKey(channel, category, "jetFakes", "nominal", f"{name}Up", "NN_score")].rebin(edges).values
        down = limit[HistKey(channel, category, "jetFakes", "nominal", f"{name}Down", "NN_score")].rebin(edges).values
        band += ((up - down) / 2) ** 2
        if name.startswith(FRACTION_SHIFTS):
            fraction_shift += up - nominal.values
    return {"T": nominal.values.tolist(), "T_stat": np.sqrt(nominal.variances).tolist(), "sigma_ff": np.sqrt(band).tolist(),
            "T_fraction_up_minus_T": fraction_shift.tolist(), "ff_shifts": len(shifts)}


def bins_of(era: str, channel: str, category: str, limit_cards: Path) -> list[float]:
    record = json.loads((limit_cards / "nn_binning.json").read_text())
    return [float(e) for e in record[channel][category]["edges"]]


def results_of(config_path: str, overrides: list[str], era: str, channels: list[str], limit_run: str) -> list[dict]:
    config = load_config(config_path, [f"era={era}", *overrides])
    analysis = load_analysis(config)
    hset = HistogramSet.load(Path(config.output_dir) / "control_shapes.root", keep=lambda key: key.region in (AR, FRACTION_REGION, f"{FRACTION_REGION}_up"))
    limit_dir = Path(limit_run.replace("{era}", era))
    limit = HistogramSet.load(limit_dir / "shapes.root", keep=lambda key: key.process == "jetFakes" and key.region == "nominal")
    results = []
    for channel in channels:
        categories = [name[len("NN_cat_"):] for name in analysis.channel(channel).variables if name.startswith("NN_cat_")]
        for category in categories:
            variable = f"NN_ar_{category}" if category.startswith(NN_SIGNAL_CLASS) else f"NN_cat_{category}"
            edges = bins_of(era, channel, category, limit_dir / "datacards" / "all")
            entry = {"era": era, "channel": channel, "category": category, "edges": edges, "truth": truth(hset, analysis, channel, variable, edges),
                     "payload": payload_fraction(hset, analysis, channel, variable, edges), "template": template(limit, channel, category, edges)}
            t, p, tpl = entry["truth"], entry["payload"], entry["template"]
            with np.errstate(divide="ignore", invalid="ignore"):
                delta = np.array(t["f_ttbar"]) - np.array(p["f_ttbar"])
                shift = np.array(tpl["T_fraction_up_minus_T"]) * delta / np.array(p["delta_f"])
                sigma = np.array(tpl["sigma_ff"])
                entry["delta_f"] = delta.tolist()
                entry["delta_f_significance"] = (delta / np.array(t["f_ttbar_error"])).tolist()
                entry["dT"] = shift.tolist()
                entry["dT_over_T"] = (shift / np.array(tpl["T"])).tolist()
                entry["dT_over_sigma_ff"] = (shift / sigma).tolist()
                entry["within_ff"] = [bool(abs(s) <= u) if np.isfinite(s) else None for s, u in zip(shift, sigma)]
            results.append(entry)
            logger.info(f"{era} {channel} {category}: |dT|/sigma_FF max {np.nanmax(np.abs(entry['dT_over_sigma_ff'])) if np.isfinite(entry['dT_over_sigma_ff']).any() else float('nan'):.2f}")
    return results


def _cell(values, fmt="{:.2f}") -> str:
    return " ".join("-" if v is None or not np.isfinite(v) else fmt.format(v) for v in values)


def summary_lines(results: list[dict]) -> list[str]:
    bins = [(r, i) for r in results for i in range(len(r["dT"]))]
    finite = [(r, i) for r, i in bins if r["within_ff"][i] is not None]
    outside = [(r, i) for r, i in finite if not r["within_ff"][i]]
    lines = ["# NV4b: fake-factor fractions per NN category", "",
             f"{len(finite)} card bins with a fraction; outside the fake-factor uncertainty (|dT| > sigma_FF): {len(outside)}", ""]
    for r, i in outside:
        lines.append(f"- {r['era']} {r['channel']} {r['category']} bin {i} [{r['edges'][i]}, {r['edges'][i + 1]}]: f_truth {r['truth']['f_ttbar'][i]:.3f} +- {r['truth']['f_ttbar_error'][i]:.3f}, "
                     f"payload {r['payload']['f_ttbar'][i]:.3f}, dT/T {r['dT_over_T'][i]:+.3f}, dT/sigma_FF {r['dT_over_sigma_ff'][i]:+.2f}")
    lines += ["", "| era | channel | category | f_ttbar truth | +- | payload | delta_f (payload shift) | dT / T | dT / sigma_FF | sigma_FF / T |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        sigma_rel = np.array(r["template"]["sigma_ff"]) / np.array(r["template"]["T"])
        lines.append(f"| {r['era']} | {r['channel']} | {r['category']} | {_cell(r['truth']['f_ttbar'], '{:.3f}')} | {_cell(r['truth']['f_ttbar_error'], '{:.3f}')} | "
                     f"{_cell(r['payload']['f_ttbar'], '{:.3f}')} | {_cell(r['payload']['delta_f'], '{:+.3f}')} | {_cell(r['dT_over_T'], '{:+.3f}')} | {_cell(r['dT_over_sigma_ff'])} | {_cell(sigma_rel, '{:.3f}')} |")
    lines += ["", "AR composition (shares of data - genuine - lepton fakes per card bin): ttbar-J / W / other J / QCD", "",
              "| era | channel | category | ttbar-J | W | other J | QCD |", "|---|---|---|---|---|---|---|"]
    for r in results:
        s = r["truth"]["shares"]
        lines.append(f"| {r['era']} | {r['channel']} | {r['category']} | {_cell(s['ttbar_J'])} | {_cell(s['W'])} | {_cell(s['other_J'])} | {_cell(s['QCD'])} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-s", "--set", dest="overrides", action="append", default=[], help="override of the run YAML (repeatable)")
    parser.add_argument("--era", action="append", required=True)
    parser.add_argument("--channels", default=",".join(TAU_CHANNELS))
    parser.add_argument("--limit-run", required=True, help="the output_dir of the fake-factor run of the limit ({era} filled): shapes.root, datacards/all/nn_binning.json")
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results = []
    for era in args.era:
        results += results_of(args.config, args.overrides, era, args.channels.split(","), args.limit_run)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fractions.json").write_text(json.dumps({"results": results}, indent=1))
    (args.output / "fractions.md").write_text("\n".join(summary_lines(results)) + "\n")
    print("\n".join(summary_lines(results)[:20]))


if __name__ == "__main__":
    main()
