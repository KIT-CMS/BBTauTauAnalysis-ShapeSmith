"""Derive the DY correction DY_NJETS_BTAG (constants.py) from the control shapes of a dilepton run.

DY data/MC per cell of n_bjets 0, 1, >= 2 (the regions btag0, btag1 and btag2, each with >= 2 jets; the row >= 2 is
measured with two b tags, btag3p has its own heavy-flavour deficit) and n_jets 2, 3, >= 4 (the bins of the control
variable n_jets): (data - the other backgrounds) / DY, the channels summed, the subtracted ttbar times --ttbar-scale.
The run must not apply the correction itself (dy_correction: none):

    shapesmith hist -c configs/sm2018_binned_v6_dilepton.yaml --control --regions all --skip-systematics
    python scripts/derive_dy_correction.py -c configs/sm2018_binned_v6_dilepton.yaml [--channels mm,ee] [--ttbar-scale 0.93]

It prints the yields of every cell and the table in the format of constants.py.
"""
import argparse

import numpy as np
from shapesmith.config import load_analysis, load_config
from shapesmith.histogram import INCLUSIVE, NOMINAL_VARIATION, HistKey, HistogramSet

BTAG_ROWS = {0: "btag0", 1: "btag1", 2: "btag2"}  # DY_NJETS_BTAG row -> dilepton region
JET_COLUMNS = ("2", "3", ">=4")


def jet_yields(hset: HistogramSet, channel: str, process: str, region: str) -> np.ndarray:
    """The yields of a process in n_jets 2, 3 and >= 4."""
    key = HistKey(channel, INCLUSIVE, process, region, NOMINAL_VARIATION, "n_jets")
    if key not in hset:
        raise SystemExit(f"{key.path} is missing: fill the b-tag regions first (hist --control --regions all)")
    h = hset[key]
    jets = (h.edges[:-1] + h.edges[1:]) / 2
    return np.array([h.values[jets == 2].sum(), h.values[jets == 3].sum(), h.values[jets >= 4].sum()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True, help="the dilepton run YAML")
    parser.add_argument("-s", "--set", action="append", default=[], help="a run YAML override, as in shapesmith")
    parser.add_argument("--channels", default="mm,ee")
    parser.add_argument("--ttbar-scale", type=float, default=1.0, help="factor on the subtracted ttbar (TT)")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    if config.switches.get("dy_correction", "none") != "none":
        raise SystemExit("the run applies the DY correction; derive the table from a run with dy_correction: none")
    analysis = load_analysis(config)
    hset = HistogramSet.load(config.output_dir / "control_shapes.root", keep=lambda key: key.variable == "n_jets" and key.variation == NOMINAL_VARIATION)
    print(f"{'n_bjets':>8} {'n_jets':>7} {'data':>12} {'other MC':>12} {'DY':>12} {'DY data/MC':>11}")
    table = {}
    for row, region in BTAG_ROWS.items():
        data, other, dy = np.zeros((3, len(JET_COLUMNS)))
        for name in args.channels.split(","):
            channel = analysis.channel(name)
            data += jet_yields(hset, name, channel.data(), region)
            dy += jet_yields(hset, name, "DY", region)
            for process in channel.backgrounds():
                if process != "DY":
                    other += jet_yields(hset, name, process, region) * (args.ttbar_scale if process == "TT" else 1.0)
        table[row] = tuple(round(float(factor), 3) for factor in (data - other) / dy)
        for column, jets in enumerate(JET_COLUMNS):
            print(f"{row if row < 2 else '>=2':>8} {jets:>7} {data[column]:12.1f} {other[column]:12.1f} {dy[column]:12.1f} {table[row][column]:11.3f}")
    print(f"\nDY_NJETS_BTAG = {table}")


if __name__ == "__main__":
    main()
