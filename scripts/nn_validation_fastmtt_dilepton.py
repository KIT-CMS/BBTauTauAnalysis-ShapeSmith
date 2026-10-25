"""NV3 in mm and em (nn_limit_spec.md 7.1): the saturated GoF of m_fastmtt and pt_fastmtt in the light-dilepton controls.

CROWN has no FastMTT friend for em and mm, so FastMTT runs here (fastmtt_refit.py, the sources of the CROWN checkout,
bitwise equal to the friend in et/mt/tt) on the events ShapeSmith selects (shapesmith.events.load: the nominal region of
smrun2_binned_v7b_dilepton_nn_validation.yaml, >= 2 jets; em and mm, the legs as an electron or muon and a muon), with
the DY table of the era (dy_correction njets_btag). Histograms per process: the nominal, the weight variations of the run
(b tag, prefiring) and the PRELIMINARY jet pseudo-shift CMS_pseudo_jes_<era> (MC: jets +-3 % with the MET propagated,
FastMTT refitted on the shifted MET; nn_validation.pseudo_shift). FastMTT takes about 1 ms per event: a simulated
process with more than --max-mc-events events keeps those with event % k == 0 (k the smallest integer that brings it
below), its weights scaled to the sum of all (the data keep every event). The cards (shapesmith.input_gof) test the shape: the
dilepton analysis has no lnN, so a free normalisation (50 % lnN on every process) and a 20 % lnN on each simulated group
let the total and the composition float.

    python scripts/nn_validation_fastmtt_dilepton.py -c configs/smrun2_binned_v7b_dilepton_nn_validation.yaml \\
        --era 2016preVFP --era 2016postVFP --era 2017 --era 2018 -o <output> --crown <CROWN checkout> --build <dir>
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np

from shapesmith import input_gof
from shapesmith.config import load_analysis, load_config
from shapesmith.events import Query, load
from shapesmith.expressions import columns_of, evaluate
from shapesmith.histogram import INCLUSIVE, NOMINAL_VARIATION, HistKey, Histogram, HistogramSet
from shapesmith.model import NOMINAL, LnN, Variable, WeightVariation

from bbtautau_shapesmith.nn_validation import PSEUDO_JES, PSEUDO_JES_NAME, VALIDATION_BINNING, _edges, pseudo_shift

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastmtt_refit import INPUTS, OUTPUTS, refit  # noqa: E402

logger = logging.getLogger("nn_validation_fastmtt_dilepton")
VARIABLES = OUTPUTS
NORM, COMPOSITION = 1.5, 1.2


def variables() -> dict[str, Variable]:
    return {name: Variable(name, name, _edges(VALIDATION_BINNING[name])) for name in VARIABLES}


def fill(values: dict[str, np.ndarray], weights: np.ndarray) -> dict[str, Histogram]:
    result = {}
    for name, variable in variables().items():
        edges = np.asarray(variable.edges)
        result[name] = Histogram.fill(edges, np.clip(values[name], edges[0], edges[-1] - 1e-6 * (edges[-1] - edges[0])), weights)
    return result


def subsample(frame, maximum: int) -> tuple[np.ndarray, int]:
    """The rows kept (event % k == 0, k = ceil(n / maximum)) and k; every row for k = 1."""
    k = max(1, int(np.ceil(len(frame) / maximum)))
    return (frame["event"].to_numpy() % np.uint64(k) == 0) if k > 1 else np.ones(len(frame), dtype=bool), k


def histograms(config, analysis, channel: str, crown: Path, build: Path, workers: int, max_mc_events: int) -> HistogramSet:
    """Every histogram of the channel in the nominal region: data, signal and backgrounds, nominal, weight variations
    and the jet pseudo-shift (MC)."""
    definition = analysis.channel(channel)
    hset = HistogramSet()
    shifts = {direction: pseudo_shift({}, 1.0 + sign * PSEUDO_JES) for direction, sign in (("Up", 1.0), ("Down", -1.0))}
    for process in definition.processes:
        if process.role == "auxiliary":
            continue
        events = load(config, analysis, Query(channel, process.name), {"event", *INPUTS, *columns_of(shift[c] for shift in shifts.values() for c in ("met", "metphi"))})
        kind = definition.kind_of(process)
        keep, k = subsample(events.frame, max_mc_events) if kind == "mc" else (np.ones(len(events.frame), dtype=bool), 1)
        frame = events.frame[keep].reset_index(drop=True)
        scale = events.weights.sum() / events.weights[keep].sum() if k > 1 else 1.0
        values = refit(frame, channel, crown, build, workers)
        logger.info(f"{analysis.era} {channel}/{process.name}: {len(frame)} of {len(events.frame)} events (k = {k}), FastMTT refitted")

        def put(variation: str, filled: dict[str, Histogram]) -> None:
            for name, h in filled.items():
                hset[HistKey(channel, INCLUSIVE, process.name, NOMINAL, variation, name)] = h

        put(NOMINAL_VARIATION, fill(values, events.weights[keep] * scale))
        for variation in definition.variations:
            if isinstance(variation, WeightVariation) and kind in variation.applies_to:
                varied = load(config, analysis, Query(channel, process.name, NOMINAL, variation))
                if len(varied.weights) != len(events.weights):
                    raise ValueError(f"{channel}/{process.name}: {variation.name} selects other events")
                put(variation.name, fill(values, varied.weights[keep] * scale))
        if kind == "mc":
            for direction, derived in shifts.items():
                shifted = frame.copy()
                for column in ("met", "metphi"):
                    shifted[column] = evaluate(frame, derived[column]).astype(np.float32)
                put(f"{PSEUDO_JES_NAME}_{analysis.era}{direction}", fill(refit(shifted, channel, crown, build, workers), events.weights[keep] * scale))
    return hset


def shape_analysis(analysis, channels: list[str]):
    """The analysis with the FastMTT variables and the free normalisation (module docstring)."""
    simulated = sorted({p.name for c in channels for p in analysis.channel(c).processes if p.role == "background"})
    lnn = (LnN(f"CMS_nv3_norm_$CHANNEL_{analysis.era}", ("*",), NORM),) + tuple(LnN(f"CMS_nv3_composition_{name}_$CHANNEL_{analysis.era}", (name,), COMPOSITION) for name in simulated)
    channels_ = {name: dataclasses.replace(analysis.channel(name), variables=variables()) if name in channels else analysis.channel(name) for name in analysis.channels}
    return dataclasses.replace(analysis, channels=channels_, lnn=lnn)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--era", action="append", required=True)
    parser.add_argument("--channels", default="em,mm")
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--crown", required=True, type=Path, help="a CROWN checkout (src/SVFit, include/SVFit)")
    parser.add_argument("--build", required=True, type=Path, help="the build directory of FastMTT")
    parser.add_argument("--toys", type=int, default=500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-mc-events", type=int, default=400000, help="events of a simulated process refitted at most (module docstring)")
    parser.add_argument("--skip-combine", action="store_true")
    parser.add_argument("--reuse-shapes", action="store_true", help="the FastMTT histograms of an earlier run (fastmtt_shapes.root in output_dir)")
    parser.add_argument("-s", "--set", action="append", default=[], help="override a configuration entry")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    channels = args.channels.split(",")
    records, combine = [], None
    for era in args.era:
        config = load_config(args.config, [f"era={era}", "switches.dy_correction=njets_btag", *args.set])
        combine = combine or config.combine
        analysis = load_analysis(config)
        shapes = config.output_dir / "fastmtt_shapes.root"
        if shapes.exists() and (args.reuse_shapes or args.skip_combine):
            hset = HistogramSet.load(shapes)
        else:
            hset = HistogramSet()
            for channel in channels:
                hset.update(histograms(config, analysis, channel, args.crown, args.build, args.workers, args.max_mc_events))
            hset.save(shapes)
        modified = shape_analysis(analysis, channels)
        targets = [input_gof.CardTarget(era, channel, NOMINAL, name) for channel in channels for name in VARIABLES]
        records += input_gof.run_cards(hset, modified, targets, args.output)
    results = input_gof.run_gof(records, combine, args.toys, args.output, args.skip_combine)
    input_gof.write_summary(results, args.output, title="NV3: saturated GoF of the FastMTT variables in em and mm (shape)")
    print((args.output / "input_gof.md").read_text())


if __name__ == "__main__":
    main()
