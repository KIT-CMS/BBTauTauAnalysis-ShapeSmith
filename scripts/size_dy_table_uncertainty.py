"""Size the DY-table nuisance CMS_htt_dy_table_<era> (v7 spec 6.7, family E; study_uncertainties.DY_TABLE) of one era
or 2016 half from the uncorrected control histograms of its dilepton run, the run the table was derived from
(bbtautau_shapesmith.dy_correction.table_uncertainty):

    python scripts/size_dy_table_uncertainty.py -c configs/smrun2_binned_v7b_dilepton.yaml -s era=2017

It prints the statistical part, the closure of each lepton flavour in btag1 + btag2 and the lnN.
"""
import argparse

from shapesmith.config import load_analysis, load_config
from shapesmith.histogram import NOMINAL_VARIATION, HistogramSet

from bbtautau_shapesmith.dy_correction import table_uncertainty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True, help="the dilepton run YAML")
    parser.add_argument("-s", "--set", action="append", default=[], help="a run YAML override, as in shapesmith (e.g. era=2017)")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    if config.switches.get("dy_correction", "none") != "none":
        raise SystemExit("the run applies the DY correction; size the table uncertainty on the run it was derived from (dy_correction: none)")
    analysis = load_analysis(config)
    hset = HistogramSet.load(config.output_dir / "control_shapes.root", keep=lambda key: key.variable == "n_jets" and key.variation == NOMINAL_VARIATION)
    result = table_uncertainty(hset, analysis)
    print(f"era {result.era}: table statistics {100 * result.stat:.2f} %")
    for name, (value, error) in result.residuals.items():
        print(f"  {name}: DY closure - 1 in btag1 + btag2 {100 * value:+.2f} +- {100 * error:.2f} %")
    print(f"  total {100 * result.total:.2f} % -> lnN {result.lnn}")


if __name__ == "__main__":
    main()
