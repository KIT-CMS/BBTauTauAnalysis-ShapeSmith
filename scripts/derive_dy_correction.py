"""Derive the DY correction DY_NJETS_BTAG (constants.py) of one era (or 2016 half) from the control shapes of its
dilepton run (bbtautau_shapesmith.dy_correction; v7 spec 6.10). The run must not apply the correction itself
(dy_correction: none):

    shapesmith hist -c <dilepton run>.yaml -s era=2017 --control --regions all --skip-systematics
    python scripts/derive_dy_correction.py -c <dilepton run>.yaml -s era=2017 [--channels mm,ee] [--ttbar-scale em]

It prints the yields of every cell and the entry of the era for constants.DY_NJETS_BTAG.
"""
import argparse

from shapesmith.config import load_analysis, load_config
from shapesmith.histogram import NOMINAL_VARIATION, HistogramSet

from bbtautau_shapesmith.dy_correction import BTAG_ROWS, derive, table_line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True, help="the dilepton run YAML")
    parser.add_argument("-s", "--set", action="append", default=[], help="a run YAML override, as in shapesmith (e.g. era=2017)")
    parser.add_argument("--channels", default="mm,ee")
    parser.add_argument("--ttbar-scale", default="1.0", help="factor on the subtracted ttbar (TT), a number or em")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    if config.switches.get("dy_correction", "none") != "none":
        raise SystemExit("the run applies the DY correction; derive the table from a run with dy_correction: none")
    analysis = load_analysis(config)
    hset = HistogramSet.load(config.output_dir / "control_shapes.root", keep=lambda key: key.variable == "n_jets" and key.variation == NOMINAL_VARIATION)
    try:
        table, scales, cells = derive(hset, analysis, args.channels.split(","), args.ttbar_scale if args.ttbar_scale == "em" else float(args.ttbar_scale))
    except KeyError as error:
        raise SystemExit(str(error)) from error
    print(f"era {analysis.era}")
    print(f"{'n_bjets':>8} {'n_jets':>7} {'data':>12} {'other MC':>12} {'DY':>12} {'DY data/MC':>11}")
    for row, region in BTAG_ROWS.items():
        print(f"{region}: ttbar times {scales[row]:.3f}")
        for cell in (c for c in cells if c.row == row):
            print(f"{row if row < 2 else '>=2':>8} {cell.column:>7} {cell.data:12.1f} {cell.other:12.1f} {cell.dy:12.1f} {cell.factor:11.3f}")
    print(f"\nDY_NJETS_BTAG entry:\n    {table_line(analysis.era, table)}")


if __name__ == "__main__":
    main()
