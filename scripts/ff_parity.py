"""Stage-1 algorithm parity of the fake-factor measurement (FM-10): ShapeSmith on jvoss's TauFakeFactors
preselection files against his SM 2018 payload. Run where /work/jvoss, /ceph and the CROWN checkout are readable,
in the LCG_108 ShapeSmith environment (TauFakeFactors' config loader needs ROOT):

    python scripts/ff_parity.py --work <dir> [--channels et,mt,tt]

1. The preselection files <PRESELECTION>/<ch>/<sample>.root become skims <work>/skims/<ch>/<sample>/ (all columns).
2. The parity analysis: every preselection sample is the process of ff_measurement of the same content, with its
   `weight` (lumi 1); the regions are jvoss's (his resolved configuration, cuts translated to pandas syntax), and in
   every region the MC and EMB weights carry TauFakeFactors' vsJet SF of its tau-ID cuts (apply_tau_id_vsJet_weight).
   The measurement is ff_measurement.legs(channel, embedding=True) with the committed tables.
3. `shapesmith measure`, then every payload is compared with jvoss's (CROWN bbtautau 1570b43): names, inputs, syst
   keys and bin edges identical, values within 1e-9 relative. Expected: the et/mt fraction n_jets edges (U4). The
   SystMCShiftDown of the corrections is an explained difference (U13: ours is smoothed at the centres of mass,
   TauFakeFactors' at the bin centres, because a deepcopy drops them), so the comparison is repeated with it refitted
   at the bin centres, as TauFakeFactors fits it (with_bin_centre_down_shifts).
4. The payloads are read with the CROWN friend's ff_payloads.read_legs.
5. --suggest-binning parity: TauFakeFactors' adjust_binning.get_binning on data.root against ShapeSmith's suggestions.
"""
from __future__ import annotations

import argparse
import copy
import glob
import gzip
import io
import json
import pickle
import re
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from shapesmith.config import NtupleConfig, RunConfig
from shapesmith.measurements import MeasureContext, run_measure
from shapesmith.measurements.fake_factors import FakeFactorMeasurement, Leg, ProcessFF
from shapesmith.measurements.fake_factors.binning import suggestions
from shapesmith.measurements.fake_factors.fit import curve, sparsify
from shapesmith.measurements.fake_factors.hist import Hist
from shapesmith.measurements.fake_factors.measure import EventSource
from shapesmith.model import Analysis, Channel, Process, Region, Sample, Selection
from shapesmith.store import skim_path, write_skim
from shapesmith.validate import validate

from bbtautau_shapesmith.ff_measurement import legs
from bbtautau_shapesmith.processes import SPLITS

FF_UPDATED = Path("/work/jvoss/FF_Updated")
CONFIG_DIR = FF_UPDATED / "configs" / "non_res_HH" / "2018"
PRESELECTION = Path("/ceph/jvoss/FFmethod/upart_17_09_26_sda_nom_emb/preselection/2018")
WORKDIR = FF_UPDATED / "workdir" / "ff_upart_17_09_26_sda_nom_emb" / "2018"  # the run of the SM payload
CROWN_ANALYSIS = Path("/work/sdaigler/bbtautau/KingMaker/CROWN/analysis_configurations/bbtautau")
REFERENCE_COMMIT = "1570b43"
PROCESSES = {  # preselection sample -> ff_measurement process
    "data": "data", "embedding": "EMB", "Wjets": "W",
    **{f"{sample}_{part}": SPLITS[group][part] for sample, group in (("DYjets", "DY"), ("ttbar", "TT"), ("ST", "ST"), ("diboson", "VV")) for part in "TLJ"},
}
EXPECTED_DIFFERENCES = {  # (channel, correction, input): (our edges, jvoss's edges, which route n_jets = 2 to the >=3 fractions; U4)
    ("et", "process_fractions", "n_jets"): ([-0.5, 2.5, 22.5], [-0.5, 1.5, 22.5]),
    ("mt", "process_fractions", "n_jets"): ([-0.5, 2.5, 22.5], [-0.5, 1.5, 22.5]),
    # the corrections share the split of their fake factors (FM-12); jvoss's tt ttbar corrections end at 22.5 jets, their
    # fake factors at 23.5 (the same events)
    **{("tt", f"{p}_non_closure_{v}_correction", "n_jets"): ([1.5, 2.5, 23.5], [1.5, 2.5, 22.5])
       for p, variables in (("ttbar", ("tau_decaymode_1", "pt_2", "met", "mt_tot", "pt_vis", "mass_1", "mass_2")),
                            ("ttbar_subleading", ("tau_decaymode_2", "pt_1", "met", "mt_tot", "pt_vis", "mass_2", "mass_1")))
       for v in variables},
}
TOLERANCE = 1e-9


def tau_fake_factors():
    """TauFakeFactors' config loader, SplitQuantities and adjust_binning (wurlitzer stubbed: log capturing only)."""
    wurlitzer = types.ModuleType("wurlitzer")
    wurlitzer.STDOUT, wurlitzer.pipes = None, None
    sys.modules.setdefault("wurlitzer", wurlitzer)
    sys.path.insert(0, str(FF_UPDATED))
    import adjust_binning
    import helper.functions as functions
    from rich.console import Console

    adjust_binning.console = Console(file=io.StringIO())  # set by its __main__ block; its tables are not needed
    return functions, adjust_binning


def kind(sample: str) -> str:
    return {"data": "data", "embedding": "embedding"}.get(sample, "mc")


def convert(channel: str, skim_dir: Path) -> None:
    """Every preselection file of the channel as one skim file (all its columns, norm_weight 1)."""
    for path in sorted((PRESELECTION / channel).glob("*.root")):
        sample = path.stem
        out = skim_path(skim_dir, channel, sample, path.name)
        if out.exists():
            continue
        frame = uproot.open(path)["ntuple"].arrays(library="pd")
        frame["sample_nick"] = pd.array([sample] * len(frame), dtype="string")
        frame["norm_weight"] = 1.0
        for column, sample_kind in (("is_data", "data"), ("is_mc", "mc"), ("is_embedding", "embedding")):
            frame[column] = kind(sample) == sample_kind
        write_skim(frame, out)
        print(f"converted {path} ({len(frame)} events)")


def pandas_syntax(cut: str) -> str:
    return re.sub(r"!(?!=)", "~", cut.replace("&&", "&").replace("||", "|"))


def tau_sf(channel: str, cut_name: str, cut: str) -> tuple[str, str]:
    """TauFakeFactors' vsJet SF of one tau-ID cut (weights.apply_tau_id_vsJet_weight with get_wps): the weight name
    and expression."""
    leg = "2" if channel != "tt" else cut_name.rsplit("_")[5]
    wps = [part.rsplit("_")[3] for part in cut.split("&&")] if "&&" in cut else [cut.rsplit("_")[3]]
    gen, iso = f"(gen_match_{leg} == 5)", lambda wp: f"id_tau_vsJet_{wp}_{leg}"
    if len(wps) == 1:
        (wp,) = wps
        expr = f"{gen} * (({iso(wp)} > 0.5) * id_wgt_tau_vsJet_{wp}_{leg} + ({iso(wp)} < 0.5)) + (gen_match_{leg} != 5)"
    else:
        loose, tight = wps
        expr = f"{gen} * (({iso(tight)} > 0.5) + ({iso(tight)} < 0.5) * ({iso(loose)} > 0.5) * id_wgt_tau_vsJet_{tight}_{leg} + ({iso(loose)} < 0.5)) + (gen_match_{leg} != 5)"
    return f"tau_id_{leg}", expr


def jvoss_regions(functions, channel: str) -> dict[str, dict[str, str]]:
    """jvoss's cuts of every region of ff_measurement.legs, by region name (C++ syntax)."""
    config = functions.load_config(str(CONFIG_DIR / f"fake_factors_{channel}.yaml"))
    corrections = functions.load_config(str(CONFIG_DIR / f"corrections_{channel}.yaml"))
    regions = {}
    for process, process_config in config["target_processes"].items():
        if process.startswith("QCD"):
            regions[f"{process}_sr_like"], regions[f"{process}_ar_like"] = process_config["SRlike_cuts"], process_config["ARlike_cuts"]
            for role, to_ar_sr in (("orthogonal", False), ("dr_sr", True)):
                modified = copy.deepcopy(config)
                functions.modify_config(config=modified, corr_config=corrections["target_processes"][process]["DR_SR"], process=process, to_AR_SR=to_ar_sr)
                regions[f"{process}_{role}_sr_like"] = modified["target_processes"][process]["SRlike_cuts"]
                regions[f"{process}_{role}_ar_like"] = modified["target_processes"][process]["ARlike_cuts"]
        else:
            regions[f"{process}_sr"], regions[f"{process}_ar"] = process_config["SR_cuts"], process_config["AR_cuts"]
            for role, cuts in (("sr_like", process_config["SRlike_cuts"]), ("ar_like", process_config["ARlike_cuts"])):
                regions[f"{process}_scale_{role}"] = cuts
                regions[f"{process}_scale_{role}_ss"] = {**cuts, "tau_pair_sign": "(q_1*q_2) > 0"}  # FF_ttbar's same-sign QCD
    for name in ("process_fractions", "process_fractions_subleading"):
        if name in config:
            regions[f"{name}_ar"] = config[name]["AR_cuts"]
    return regions


def parity_channel(functions, channel: str, measurement: FakeFactorMeasurement) -> Channel:
    samples = tuple(Sample(path.stem, path.stem, kind(path.stem)) for path in sorted((PRESELECTION / channel).glob("*.root")))
    tau_legs = ("1", "2") if channel == "tt" else ("2",)
    processes = tuple(
        Process(PROCESSES[s.nick], s.nick, "data" if s.kind == "data" else "background", s.nick,
                Selection(weights={"weight": "weight", **({f"tau_id_{leg}": "1.0" for leg in tau_legs} if s.kind != "data" else {})}))
        for s in samples
    )
    regions, cut_names = [], set()
    for name, cuts in jvoss_regions(functions, channel).items():
        sfs = dict(tau_sf(channel, cut_name, cut) for cut_name, cut in cuts.items() if cut_name.startswith("had_tau_id_vs_jet"))
        regions.append(Region(name, replace_cuts={n: pandas_syntax(c) for n, c in cuts.items()}, replace_weights=sfs))
        cut_names |= set(cuts)
    keep = {r.name for r in regions} & {name for leg in measurement.legs[channel] for name in _region_names(leg)}
    return Channel(channel, samples, {}, {name: "True" for name in sorted(cut_names)}, processes, tuple(r for r in regions if r.name in keep))


def _region_names(leg) -> set[str]:
    names = {leg.fractions.region}
    for process in leg.processes:
        names |= {process.sr_like, process.ar_like}
        if process.dr_sr is not None:
            names |= {process.dr_sr.sr_like, process.dr_sr.ar_like, process.dr_sr.sr, process.dr_sr.ar}
        if process.scale is not None:
            s = process.scale
            names |= {s.sr_like, s.ar_like, s.sr_like_same_sign, s.ar_like_same_sign}
    return names


def in_tau_fake_factors_order(leg: Leg, channel: str) -> Leg:
    """The leg with every subtraction in the order of TauFakeFactors' sample list (the glob order of the preselection
    directory): the centre of mass of a subtraction is clipped into the bin at every step, so it depends on the order."""
    order = [PROCESSES[Path(path).stem] for path in glob.glob(str(PRESELECTION / channel / "*.root"))]

    def ordered(names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(names, key=order.index))

    def process(p: ProcessFF) -> ProcessFF:
        dr_sr = replace(p.dr_sr, subtract=ordered(p.dr_sr.subtract)) if p.dr_sr else None
        scale = replace(p.scale, subtract=ordered(p.scale.subtract)) if p.scale else None
        return replace(p, subtract=ordered(p.subtract), dr_sr=dr_sr, scale=scale)

    return replace(leg, qcd=process(leg.qcd), ttbar=process(leg.ttbar), fractions=replace(leg.fractions, subtract=ordered(leg.fractions.subtract)))


def build(config: RunConfig) -> Analysis:
    functions, _ = tau_fake_factors()
    measurement = FakeFactorMeasurement({channel: tuple(in_tau_fake_factors_order(leg, channel) for leg in legs(channel, embedding=True)) for channel in config.channels})
    channels = {channel: parity_channel(functions, channel, measurement) for channel in config.channels}
    return Analysis("ff_parity_jvoss_preselection", "2018", 1.0, None, channels, measurement=measurement)


def compare_nodes(ours, theirs, path: str, report: dict, allowed_edges: dict) -> None:
    """Every difference of two payload nodes, into report["problems"]; the largest relative value difference."""
    if isinstance(theirs, (int, float)) or isinstance(ours, (int, float)):
        if not isinstance(ours, (int, float)) or not isinstance(theirs, (int, float)):
            report["problems"].append(f"{path}: {type(ours).__name__} vs {type(theirs).__name__}")
            return
        difference = abs(ours - theirs) / max(abs(ours), abs(theirs), 1e-12)
        report["max_relative"] = max(report["max_relative"], difference)
        if difference > TOLERANCE:
            report["problems"].append(f"{path}: {ours!r} vs {theirs!r}")
        return
    if ours.get("nodetype") != theirs.get("nodetype") or ours.get("input") != theirs.get("input"):
        report["problems"].append(f"{path}: node {ours.get('nodetype')}({ours.get('input')}) vs {theirs.get('nodetype')}({theirs.get('input')})")
        return
    if ours["nodetype"] == "binning":
        expected, reference = allowed_edges.get(ours["input"], (theirs["edges"], theirs["edges"]))
        if theirs["edges"] != reference:
            report["problems"].append(f"{path}: jvoss's edges {theirs['edges']} are not the known {reference}")
        if ours["edges"] != expected:
            report["edges_equal"] = False
            if len(ours["edges"]) != len(expected) or not np.allclose(ours["edges"], expected, rtol=TOLERANCE, atol=0):
                report["problems"].append(f"{path}: edges {ours['edges'][:6]}... vs {expected[:6]}... ({len(ours['edges'])} vs {len(expected)})")
                return
        if ours["input"] in allowed_edges:
            report["expected"].append(f"{path}: {ours['input']} edges {ours['edges']} (jvoss {theirs['edges']})")
        for index, (a, b) in enumerate(zip(ours["content"], theirs["content"])):
            compare_nodes(a, b, f"{path}[{index}]", report, allowed_edges)
        return
    keys_ours = {item["key"]: item["value"] for item in ours["content"]}
    keys_theirs = {item["key"]: item["value"] for item in theirs["content"]}
    if set(keys_ours) != set(keys_theirs):
        report["problems"].append(f"{path}: keys only here {sorted(set(keys_ours) - set(keys_theirs))}, only in jvoss's {sorted(set(keys_theirs) - set(keys_ours))}")
    for key in sorted(set(keys_ours) & set(keys_theirs)):
        compare_nodes(keys_ours[key], keys_theirs[key], f"{path}/{key}", report, allowed_edges)
    if (ours.get("default") is None) != (theirs.get("default") is None):
        report["problems"].append(f"{path}: default present {ours.get('default') is not None} vs {theirs.get('default') is not None}")
    elif ours.get("default") is not None:
        compare_nodes(ours["default"], theirs["default"], f"{path}/default", report, allowed_edges)


def reference_payload(name: str) -> dict:
    blob = subprocess.run(["git", "-C", str(CROWN_ANALYSIS), "show", f"{REFERENCE_COMMIT}:payloads/fake_factors/sm/2018/{name}"], capture_output=True, check=True).stdout
    return json.loads(gzip.decompress(blob))


def read_payload(output: Path, name: str) -> dict:
    return json.loads(gzip.decompress((output / name).read_bytes()))


def corrections_by_record_key(measurement: FakeFactorMeasurement, channel: str) -> dict:
    """The data-driven corrections (they have a SystMCShift) by their key in measurement_<ch>.json."""
    binned = {}
    for leg in measurement.legs[channel]:
        name = leg.qcd.name + leg.suffix
        binned.update({f"{name}_non_closure_{b.variable}": b for b in leg.qcd.non_closures})
        binned[f"{name}_DR_SR"] = leg.qcd.dr_sr.correction
    return binned


def bin_centre_down_shift(entry: dict, fit) -> np.ndarray | None:
    """TauFakeFactors' SystMCShiftDown of a correction at full resolution, None where it equals ours. TauFakeFactors
    deep-copies the down-shifted histograms of its corrections (non_closure_correction, DR_SR_correction); a deep copy
    loses the centre-of-mass attributes, so its SystMCShiftDown is smoothed at the bin centres."""
    if fit.kind == "binwise" or entry["reset"]:
        return None
    down = entry["mc_shifted"][1]
    edges, errors = np.array(entry["edges"]), np.array(down["errors"])
    centres = 0.5 * (edges[:-1] + edges[1:])
    return curve(Hist(edges, np.array(down["y"]), errors**2, np.ones_like(errors), centres, np.array(down["y"]), errors, errors), fit, 1.0).nominal


def with_bin_centre_down_shifts(corrections: dict, record: dict, measurement: FakeFactorMeasurement, channel: str) -> dict:
    """The corrections payload with every smoothed SystMCShiftDown as TauFakeFactors fits it (bin_centre_down_shift)."""
    binned = corrections_by_record_key(measurement, channel)
    result = copy.deepcopy(corrections)
    for correction in result["corrections"]:
        key = correction["name"].removesuffix("_correction")
        for item in correction["data"]["content"]:
            if key not in binned or not item["key"].endswith("_CorrSystMCShiftDown"):
                continue
            for category, (node, entry) in enumerate(zip(item["value"]["content"], record[key])):
                down = bin_centre_down_shift(entry, binned[key].fits[category])
                if down is not None:  # stored on the grid of the nominal curve, as every variation
                    node["edges"], node["content"] = (a.tolist() for a in sparsify(np.array(entry["curve"]["edges"]), down))
    return result


def compare(output: Path, channel: str, overrides: dict | None = None) -> dict:
    """Our payloads (or `overrides` by file name) against jvoss's."""
    report = {"problems": [], "expected": [], "max_relative": 0.0, "edges_equal": True, "corrections": 0}
    for name in (f"fake_factors_{channel}.json.gz", f"FF_corrections_{channel}.json.gz"):
        ours, theirs = (overrides or {}).get(name) or read_payload(output, name), reference_payload(name)
        by_name = {c["name"]: c for c in ours["corrections"]}
        reference = {c["name"]: c for c in theirs["corrections"]}
        if set(by_name) != set(reference):
            report["problems"].append(f"{name}: corrections only here {sorted(set(by_name) - set(reference))}, only in jvoss's {sorted(set(reference) - set(by_name))}")
        for correction in sorted(set(by_name) & set(reference)):
            a, b = by_name[correction], reference[correction]
            if [(v["name"], v["type"]) for v in a["inputs"]] != [(v["name"], v["type"]) for v in b["inputs"]]:
                report["problems"].append(f"{correction}: inputs {[v['name'] for v in a['inputs']]} vs {[v['name'] for v in b['inputs']]}")
            allowed = {key[2]: edges for key, edges in EXPECTED_DIFFERENCES.items() if key[:2] == (channel, correction)}
            compare_nodes(a["data"], b["data"], correction, report, allowed)
            report["corrections"] += 1
        compounds = {c["name"]: (c["stack"], [v["name"] for v in c["inputs"]]) for c in ours.get("compound_corrections") or []}
        reference_compounds = {c["name"]: (c["stack"], [v["name"] for v in c["inputs"]]) for c in theirs.get("compound_corrections") or []}
        if compounds != reference_compounds:
            report["problems"].append(f"{name}: compound corrections {compounds} vs {reference_compounds}")
    return report


def curve_parity(output: Path, channel: str, measurement: FakeFactorMeasurement) -> dict:
    """ShapeSmith's full-resolution fits (measurement_<ch>.json) against the ones TauFakeFactors pickled for its plots
    (workdir .../fake_factors|corrections/<ch>/pickle): measured points and every curve, bitwise or relative; the
    SystMCShiftDown of the corrections as TauFakeFactors fits it (bin_centre_down_shift)."""
    record = json.loads((output / f"measurement_{channel}.json").read_text())
    corrections = corrections_by_record_key(measurement, channel)
    report = {"arrays": 0, "bitwise": 0, "max_relative": 0.0, "different": []}
    pickles = sorted(WORKDIR.glob(f"*/{channel}/pickle/data_corr_*.pickle")) + sorted(WORKDIR.glob(f"*/{channel}/pickle/data_ff_*.pickle"))
    for path in pickles:
        name, label = re.match(r"data_(?:corr_|ff_\w+?_(?=QCD|ttbar))(.+)_n_jets_(.+)\.pickle", path.name).groups()
        fake_factors = path.name.startswith("data_ff_")
        orthogonal = name.endswith("_for_DRtoSR") or (fake_factors and path.parts[-4] == "corrections")
        key = ("for_DRtoSR/" if orthogonal else "") + name.removesuffix("_for_DRtoSR") + ("_fake_factors" if fake_factors else "")
        pickled = pickle.loads(path.read_bytes())
        points, curve = (pickled["ff_ratio"], pickled["variations"]) if "ff_ratio" in pickled else (pickled["corr_hist"], pickled["corr_graph"])
        entries = [e for e in record.get(key, []) if _category_label(e["split"]) == label]
        if len(entries) != 1:
            report["different"].append(f"{path.name}: no record {key} {label}")
            continue
        (entry,) = entries
        ours_down = bin_centre_down_shift(entry, corrections[key].fits[entry["category"]]) if key in corrections else None
        if ours_down is not None:
            entry = {**entry, "curve": {**entry["curve"], "SystMCShiftDown": ours_down}}
        arrays = {"x": (entry["x"], points[0]), "y": (entry["y"], points[1]), "errors": (entry["errors"], points[3][1]),
                  "edges": (entry["curve"]["edges"], curve["edges"]), "nominal": (entry["curve"]["nominal"], curve["nominal"]),
                  **{k: (entry["curve"][k], v) for k, v in curve["variations"].items()}}
        for field, (ours, theirs) in arrays.items():
            ours, theirs = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
            report["arrays"] += 1
            if ours.shape != theirs.shape:
                report["different"].append(f"{key} {label} {field}: shape {ours.shape} vs {theirs.shape}")
                continue
            if np.array_equal(ours, theirs):
                report["bitwise"] += 1
                continue
            relative = float(np.max(np.abs(ours - theirs) / np.maximum(np.abs(theirs), 1e-12)))
            report["max_relative"] = max(report["max_relative"], relative)
            report["different"].append(f"{key} {label} {field}: max relative {relative:.3g}")
    return report


def _category_label(split: list[float]) -> str:
    """TauFakeFactors' label of an n_jets category."""
    low, high = split
    if high - low == 1:
        return f"=={int(low + 0.5)}"
    return f">={int(low + 0.5)}" if low > 0 else f"<={int(high - 0.5)}"


def binning_parity(functions, adjust_binning, config: RunConfig, analysis: Analysis, channel: str) -> list[str]:
    """TauFakeFactors' adjust_binning.get_binning on data.root against ShapeSmith's suggestions, per quantity."""
    ff_config = functions.load_config(str(CONFIG_DIR / f"fake_factors_{channel}.yaml"))
    corrections = functions.load_config(str(CONFIG_DIR / f"corrections_{channel}.yaml"))
    frame = uproot.open(PRESELECTION / channel / "data.root")["ntuple"].arrays(library="pd")
    frame[frame.select_dtypes(include=["bool"]).columns] = frame.select_dtypes(include=["bool"]).astype(int)

    def cuts(*sources: dict) -> str:
        merged = {}
        for source in sources:
            merged.update(source)
        return pandas_syntax(" & ".join(f"({c})" for c in merged.values()))

    def reference(quantity: dict, region: str) -> list[list[float]]:
        bins, _, _ = adjust_binning.get_binning(df=frame, binning_config=quantity["equipopulated_binning_options"], process_cuts=region,
                                                category_splits=quantity["split_categories"], var_dependence=quantity["var_dependence"])
        return [list(edges) for edges in bins.values()]

    expected = {}
    for process, process_config in ff_config["target_processes"].items():
        sr_like = process_config["SRlike_cuts"]
        expected[f"{process}_fake_factors"] = reference(process_config, cuts(sr_like))
        correction = corrections["target_processes"][process]
        for variable, nc in correction.get("non_closure", {}).items():
            if "equipopulated_binning_options" in nc:
                expected[f"{process}_non_closure_{variable}"] = reference(nc, cuts(sr_like))
        if "DR_SR" in correction:
            dr_sr = correction["DR_SR"]
            expected[f"{process}_DR_SR"] = reference(dr_sr, cuts(sr_like, dr_sr["SRlike_cuts"]))
    for name in ("process_fractions", "process_fractions_subleading"):
        if name in ff_config:
            expected[name] = reference(ff_config[name], cuts(ff_config[name]["AR_cuts"]))
    context = MeasureContext(config, analysis, [channel], config.output_dir)
    ours = {}
    for name, category, suggested, _ in suggestions(analysis.measurement, EventSource(context, channel, analysis.measurement.columns(channel))):
        ours.setdefault(name, []).append(suggested)
    problems = [f"{name}: ShapeSmith {ours.get(name)} vs TauFakeFactors {edges}" for name, edges in expected.items() if ours.get(name) != edges]
    problems += [f"{name}: no TauFakeFactors reference" for name in set(ours) - set(expected)]
    print(f"{channel}: --suggest-binning compared for {len(expected)} quantities, {len(problems)} differences")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--channels", default="et,mt,tt")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    channels = args.channels.split(",")
    for channel in channels:
        convert(channel, args.work / "skims")
    config = RunConfig(analysis="ff_parity:build", era="2018", channels=channels, ntuples=NtupleConfig(base=str(PRESELECTION)),
                       skim_dir=args.work / "skims", output_dir=args.work / "output", workers=args.workers)
    analysis = build(config)
    validate(analysis)
    output = run_measure(config, analysis, channels)
    sys.path.insert(0, str(CROWN_ANALYSIS.parents[1]))
    from analysis_configurations.bbtautau.ff_payloads import read_legs

    functions, adjust_binning = tau_fake_factors()
    summary = {}
    for channel in channels:
        report = compare(output, channel)
        record = json.loads((output / f"measurement_{channel}.json").read_text())
        name = f"FF_corrections_{channel}.json.gz"
        explained = compare(output, channel, {name: with_bin_centre_down_shifts(read_payload(output, name), record, analysis.measurement, channel)})
        report["with_bin_centre_down_shifts"] = {k: explained[k] for k in ("problems", "max_relative", "edges_equal")}
        report["curves"] = curve_parity(output, channel, analysis.measurement)
        read = read_legs(str(output), channel)
        report["read_legs"] = {leg.name: sum(len(c.shift_keys) for c in leg.corrections) for leg in read}
        report["binning"] = binning_parity(functions, adjust_binning, config, analysis, channel)
        summary[channel] = report
        print(f"{channel}: {report['corrections']} corrections, {len(report['problems'])} problems, max relative difference {report['max_relative']:.3g}, "
              f"edges identical {report['edges_equal']}, expected differences {len(report['expected'])}, binning differences {len(report['binning'])}")
        for problem in report["problems"][:40]:
            print(f"  {problem}")
        print(f"{channel}: with TauFakeFactors' bin-centre SystMCShiftDown of the corrections: {len(explained['problems'])} problems, "
              f"max relative difference {explained['max_relative']:.3g}, edges identical {explained['edges_equal']}")
        for problem in explained["problems"][:40]:
            print(f"  {problem}")
        curves = report["curves"]
        print(f"{channel}: fits against TauFakeFactors' pickled curves: {curves['bitwise']}/{curves['arrays']} arrays bitwise, max relative difference "
              f"{curves['max_relative']:.3g}")
        for difference in curves["different"][:60]:
            print(f"  {difference}")
    (args.work / f"parity_{args.channels.replace(',', '_')}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
