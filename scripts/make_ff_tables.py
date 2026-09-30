"""Write src/bbtautau_shapesmith/ff_tables.py: the fake-factor tables (binning, fits, bandwidths, --suggest-binning
options) of jvoss's SM 2018 TauFakeFactors configuration, resolved by TauFakeFactors itself (load_config and
SplitQuantities: defaults such as the bandwidth (max - min) / 5, templates, per-category values). Run once, by hand,
where /work/jvoss/FF_Updated is readable (LCG_108 has ROOT and ruamel.yaml):

    source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
    python scripts/make_ff_tables.py

The only change against the configuration: the et/mt fraction split uses the n_jets edges [-0.5, 2.5, 22.5] of its
categories <=2 and >=3 (user decision U4; jvoss's edges [-0.5, 1.5, 22.5] route n_jets = 2 to the >=3 fractions).
"""
import ast
import hashlib
import sys
import types
from pathlib import Path

FF_UPDATED = Path("/work/jvoss/FF_Updated")
CONFIG_DIR = FF_UPDATED / "configs" / "non_res_HH" / "2018"
OUTPUT = Path(__file__).resolve().parents[1] / "src" / "bbtautau_shapesmith" / "ff_tables.py"
CHANNELS = ("et", "mt", "tt")
FRACTION_EDGES = {"et": (-0.5, 2.5, 22.5), "mt": (-0.5, 2.5, 22.5)}  # U4


def import_tau_fake_factors():
    wurlitzer = types.ModuleType("wurlitzer")  # imported by ff_functions for log capturing only
    wurlitzer.STDOUT, wurlitzer.pipes = None, None
    sys.modules["wurlitzer"] = wurlitzer
    sys.path.insert(0, str(FF_UPDATED))
    import helper.ff_functions as ff_functions
    import helper.functions as functions

    return functions, ff_functions


def text(value: str) -> str:
    return f'"{value}"'


def sequence(items: list[str]) -> str:
    return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"


def fit(option: str, bandwidth: float) -> str:
    """TauFakeFactors' fit option as a Fit: binwise, smoothed or `binwise#[i,...]+smoothed` (_get_index_and_slices)."""
    if option == "binwise":
        return 'Fit("binwise")'
    if option == "smoothed":
        return f'Fit("smoothed", {float(bandwidth)!r})'
    groups = [ast.literal_eval(group) for group in option.split("+")[0].split("#")[1:]]
    if len(groups) != 1:
        raise ValueError(f"unsupported fit option {option}")
    last = groups[0][-1]
    side = f"binwise_left={last + 1}" if last >= 0 else f"binwise_right={-last}"
    return f'Fit("smoothed", {float(bandwidth)!r}, {side})'


def n_bins(options: dict, category: str) -> int:
    value = options["var_dependence_n_bins"]
    return value if isinstance(value, int) else value[category]


def equipopulated(config: dict, categories: list[str]) -> str:
    options = config["equipopulated_binning_options"]
    variable = options["variable_config"][config["var_dependence"]]
    counts = tuple(n_bins(options, category) for category in categories)
    add_left = tuple(float(edge) for edge in options.get("add_left", ()))
    extra = f", add_left={add_left!r}" if add_left else ""
    return f"Equipopulated({counts!r}, {float(variable['min'])!r}, {float(variable['max'])!r}, {variable.get('rounding', 2)!r}{extra})"


def binned(ff_functions, config: dict, option_key: str | None) -> str:
    """Binned(...) of one TauFakeFactors quantity, per category of its split."""
    splits = list(ff_functions.SplitQuantities(config))
    categories = [split.split["n_jets"] for split in splits]
    edges = tuple(tuple(float(edge) for edge in split.var_bins) for split in splits)
    parts = [text(config["var_dependence"]), repr(edges)]
    if option_key is not None:
        parts.append(sequence([fit(getattr(split, option_key), split.bandwidth) for split in splits]))
    if "equipopulated_binning_options" in config:
        parts.append(("" if option_key is not None else "(), ") + equipopulated(config, categories))
    return f"Binned({', '.join(parts)})"


def split(config: dict, edges: tuple[float, ...] | None = None) -> str:
    (variable, config_edges), = config["split_categories_binedges"].items()
    return f"Split({text(variable)}, {tuple(float(e) for e in (edges or config_edges))!r})"


def channel_table(functions, ff_functions, channel: str) -> list[str]:
    fake_factors = functions.load_config(str(CONFIG_DIR / f"fake_factors_{channel}.yaml"))
    corrections = functions.load_config(str(CONFIG_DIR / f"corrections_{channel}.yaml"))
    lines = [f"    {text(channel)}: {{"]
    for process, config in fake_factors["target_processes"].items():
        correction = corrections["target_processes"][process]
        non_closures = [binned(ff_functions, c, "correction_option") for c in correction.get("non_closure", {}).values()]
        lines += [f"        {text(process)}: {{", f'            "split": {split(config)},', f'            "fake_factors": {binned(ff_functions, config, "fit_option")},']
        lines += ['            "non_closures": ('] + [f"                {b}," for b in non_closures] + ["            ),"]
        if "DR_SR" in correction:
            dr_sr = correction["DR_SR"]
            lines.append(f'            "dr_sr": {binned(ff_functions, dr_sr, "correction_option")},')
            lines += ['            "dr_sr_non_closures": ('] + [f'                {binned(ff_functions, c, "correction_option")},' for c in dr_sr["non_closure"].values()] + ["            ),"]
        lines.append("        },")
    for name in ("process_fractions", "process_fractions_subleading"):
        if name in fake_factors:
            config = fake_factors[name]
            lines.append(f'        {text(name)}: {{"split": {split(config, FRACTION_EDGES.get(channel))}, "binned": {binned(ff_functions, config, None)}}},')
    return lines + ["    },"]


def main() -> None:
    functions, ff_functions = import_tau_fake_factors()
    hashes = [f"#   {path.name} {hashlib.sha256(path.read_bytes()).hexdigest()}" for path in sorted(CONFIG_DIR.glob("*.yaml"))]
    lines = [
        '"""The fake-factor tables of the SM 2018 measurement: per channel and TauFakeFactors process the split, the binning,',
        "fits and --suggest-binning options of the fake factors, the non-closures (in their order) and the DR->SR correction,",
        "and of the process fractions.",
        "",
        f"Generated by scripts/make_ff_tables.py from TauFakeFactors' resolved configuration {CONFIG_DIR}; do not",
        "edit. One change against it: the et/mt fraction split has the n_jets edges [-0.5, 2.5, 22.5] of its categories",
        '<=2 and >=3 (U4).',
        '"""',
        "# Configuration files (sha256):",
        *hashes,
        "from shapesmith.measurements.fake_factors import Binned, Equipopulated, Fit, Split",
        "",
        "TABLES = {",
    ]
    for channel in CHANNELS:
        lines += channel_table(functions, ff_functions, channel)
    OUTPUT.write_text("\n".join(lines + ["}", ""]))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
