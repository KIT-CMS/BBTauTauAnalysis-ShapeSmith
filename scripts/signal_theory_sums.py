"""Measure the inclusive normalisations of the theory weights of samples from the Runs trees of their NanoAOD (v8 spec
SH20, S6, P8) and write the table of the analysis from them (bbtautau_shapesmith.theory_norms.RUNS_NORM):

    source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh      # uproot, XRootD
    export X509_USER_PROXY=/tmp/x509up_u12108
    # measure: the samples of a sample type of an era (or --nick <nick>, or every era without --era)
    python scripts/signal_theory_sums.py --sample-type hh2b2tau --sample-type ttbar --era 2018 --json sums_2018.json
    # the table module from one or more measurements (later files replace the samples of earlier ones)
    python scripts/signal_theory_sums.py --table runs_sums_norm.json [sums_2018.json ...] > src/bbtautau_shapesmith/theory_norms.py

The n-tuples hold only the selected events, so the inclusive sums cannot come from them. NanoAOD stores LHEScaleSumw,
PSSumw and LHEPdfSumw per Runs entry divided by the genEventSumw of the entry; the normalisation of a weight w_i of a
sample is f_i = sum_e genEventSumw_e XSumw_e[i] / sum_e genEventSumw_e = sum(genWeight w_i) / sum(genWeight) over every
generated event. A varied weight is normalised before the acceptance as w_i / f_i (TOP recommendation), so that its
template keeps the inclusive rate. The files of a sample are those of DAS (dasgoclient), read over xrootd.

The JSON of a measurement has per DAS name the fields nick, era, sample_type, genEventSumw, genEventCount, scale (the
nine LHEScaleWeight entries by name), ps (the four PSWeight entries) and pdf (n and all: every LHEPdfWeight member);
the measurement of 2026-11-24 over all files of the 20 signal and 17 ttbar samples of the four eras is
docs/analysen/v8/v8_w0_files/B/runs_sums_norm.json (W0, part B).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATABASE = REPO.parent / "KingMaker_sample_database" / "nanoAOD_v15" / "datasets.json"
REDIRECTOR = "root://cms-xrd-global.cern.ch/"
DASGOCLIENT = "/cvmfs/cms.cern.ch/common/dasgoclient"
# the entries of the NanoAOD weight vectors by name (branch titles checked for every MC nick in W0)
SCALE_KEYS = ("muR0.5_muF0.5", "muR0.5_muF1", "muR0.5_muF2", "muR1_muF0.5", "muR1_muF1", "muR1_muF2", "muR2_muF0.5", "muR2_muF1", "muR2_muF2")
PS_KEYS = ("isr_murfac2", "fsr_murfac2", "isr_murfac0.5", "fsr_murfac0.5")
SUMS = {"LHEScaleSumw": len(SCALE_KEYS), "PSSumw": len(PS_KEYS)}  # the expected lengths; LHEPdfSumw has 101 or 103
# the varied weights of the analysis: the CROWN shift with its direction -> where its normalisation is in a measurement
# (QCDscale_ren/fac: muR and muF times 2 Up and 0.5 Down; ps_isr/fsr: the scale times 2 Up and 0.5 Down)
NORMS = {
    "QCDscale_renUp": ("scale", "muR2_muF1"), "QCDscale_renDown": ("scale", "muR0.5_muF1"),
    "QCDscale_facUp": ("scale", "muR1_muF2"), "QCDscale_facDown": ("scale", "muR1_muF0.5"),
    "ps_isrUp": ("ps", "isr_murfac2"), "ps_isrDown": ("ps", "isr_murfac0.5"),
    "ps_fsrUp": ("ps", "fsr_murfac2"), "ps_fsrDown": ("ps", "fsr_murfac0.5"),
}
PDF_MEMBERS = slice(1, 101)  # the 100 Hessian eigenvectors (member 0 is the central one, 101 and 102 alpha_s)
PDF_SAMPLE_TYPES = ("hh2b2tau",)  # the sample types whose n-tuples have the eigenvector weights (V27: the signal only)
DIGITS = 6


def chosen(database: dict, sample_types: list[str], nicks: list[str], era: str | None) -> dict[str, dict]:
    """nick -> database entry of the samples asked for: those of the sample types (of the era) and the named nicks."""
    unknown = [nick for nick in nicks if nick not in database]
    if unknown:
        raise SystemExit(f"not in the sample database: {', '.join(unknown)}")
    by_type = {nick: entry for nick, entry in database.items() if entry["sample_type"] in sample_types and (era is None or entry["era"] == era)}
    return {**by_type, **{nick: database[nick] for nick in nicks}}


def files(dbs: str) -> list[str]:
    output = subprocess.run([DASGOCLIENT, f"-query=file dataset={dbs}"], check=True, capture_output=True, text=True).stdout
    return sorted(line for line in output.split() if line.endswith(".root"))


def combine(entries: list[dict[str, np.ndarray]]) -> dict:
    """The normalisations of one sample from its Runs entries (each with genEventSumw, genEventCount and the sums
    divided by genEventSumw): the genEventSumw-weighted means of every entry of LHEScaleSumw, PSSumw and LHEPdfSumw."""
    total = sum(float(entry["genEventSumw"]) for entry in entries)
    count = sum(int(entry["genEventCount"]) for entry in entries)
    means = {}
    for key in (*SUMS, "LHEPdfSumw"):
        lengths = {len(entry[key]) for entry in entries}
        if len(lengths) != 1 or (key in SUMS and lengths != {SUMS[key]}):
            raise ValueError(f"{key}: lengths {sorted(lengths)} in the Runs entries" + (f", expected {SUMS[key]}" if key in SUMS else ""))
        means[key] = sum(float(entry["genEventSumw"]) * np.asarray(entry[key], dtype=float) for entry in entries) / total
    return {
        "genEventSumw": total, "genEventCount": count,
        "scale": dict(zip(SCALE_KEYS, means["LHEScaleSumw"].tolist())),
        "ps": dict(zip(PS_KEYS, means["PSSumw"].tolist())),
        "pdf": {"n": len(means["LHEPdfSumw"]), "all": means["LHEPdfSumw"].tolist()},
    }


def runs_entries(urls: list[str]) -> list[dict[str, np.ndarray]]:
    """Every Runs entry of the files."""
    import uproot  # with XRootD: only the measurement needs it

    entries = []
    for url in urls:
        with uproot.open(url, timeout=600) as handle:
            runs = handle["Runs"].arrays(["genEventSumw", "genEventCount", *SUMS, "LHEPdfSumw"], library="np")
        entries += [{key: values[index] for key, values in runs.items()} for index in range(len(runs["genEventSumw"]))]
    return entries


def measure(samples: dict[str, dict]) -> dict:
    """The measurement of the samples, keyed by their DAS name."""
    result = {}
    for nick, entry in samples.items():
        urls = [REDIRECTOR + path for path in files(entry["dbs"])]
        sums = combine(runs_entries(urls))
        if sums["genEventCount"] != entry["nevents"]:
            raise SystemExit(f"{nick}: {sums['genEventCount']} generated events in the Runs trees, {entry['nevents']} in the database")
        result[entry["dbs"]] = {"dbs": entry["dbs"], "nick": nick, "era": entry["era"], "sample_type": entry["sample_type"], "files": len(urls), **sums}
        print(f"{nick}: {len(urls)} files, genEventSumw {sums['genEventSumw']:.6g}, muR 2 / 0.5 {sums['scale']['muR2_muF1']:.4f} / {sums['scale']['muR0.5_muF1']:.4f}", file=sys.stderr)
    return result


def norm_of(entry: dict) -> dict:
    """The table entry of one measured sample: its era, the normalisation of every varied weight (NORMS) and, for the
    sample types with the eigenvector weights, of the 100 eigenvectors."""
    result = {"era": entry["era"], "sample_type": entry["sample_type"], **{name: round(entry[block][key], DIGITS) for name, (block, key) in NORMS.items()}}
    if entry["sample_type"] in PDF_SAMPLE_TYPES:
        members = entry["pdf"]["all"][PDF_MEMBERS]
        if len(members) != 100:
            raise ValueError(f"{entry['nick']}: {len(members)} PDF eigenvectors")
        result["pdf"] = tuple(round(value, DIGITS) for value in members)
    return result


def table(measurements: list[dict]) -> dict[str, dict]:
    """nick -> table entry of every sample of the measurements; a later measurement replaces a sample of an earlier one."""
    return {entry["nick"]: norm_of(entry) for measurement in measurements for entry in measurement.values()}


HEADER = '''"""The inclusive normalisations of the theory weights per sample (v8 spec SH20): for a varied weight w of a sample,
sum(genWeight w) / sum(genWeight) over every generated event, from the Runs trees of its NanoAOD.

Generated by scripts/signal_theory_sums.py --table from {sources}; do not edit. Per nick: the era, the sample type,
the normalisation of the CROWN shifts QCDscale_ren, QCDscale_fac (muR and muF times 2 Up, 0.5 Down), ps_isr and ps_fsr
(the scale times 2 Up, 0.5 Down) in both directions and, for the signal sample type, of the 100 Hessian eigenvectors
lhe_pdf_weight_1..100 (`pdf`). A varied weight over its normalisation keeps the inclusive rate of the sample
(systematics.theory_norm).
"""
'''


def module(measurements: list[dict], sources: list[str]) -> str:
    """The text of theory_norms.py: one entry per nick, sorted, the eigenvectors ten to a line."""
    lines = [HEADER.format(sources=", ".join(sources)) + "RUNS_NORM = {"]
    for nick, entry in sorted(table(measurements).items()):
        lines.append(f"    {json.dumps(nick)}: {{")
        lines.append("        " + ", ".join(f"{json.dumps(key)}: {json.dumps(entry[key])}" for key in ("era", "sample_type")) + ",")
        for shift in sorted({name.removesuffix("Up").removesuffix("Down") for name in NORMS}):
            lines.append("        " + ", ".join(f"{json.dumps(shift + direction)}: {entry[shift + direction]!r}" for direction in ("Up", "Down")) + ",")
        if "pdf" in entry:
            lines.append('        "pdf": (')
            lines += ["            " + ", ".join(repr(value) for value in entry["pdf"][start:start + 10]) + "," for start in range(0, len(entry["pdf"]), 10)]
            lines.append("        ),")
        lines.append("    },")
    return "\n".join(lines) + "\n}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", type=Path, nargs="+", metavar="JSON", help="print the module theory_norms.py from these measurements instead of measuring")
    parser.add_argument("--sample-type", action="append", default=[], help="measure every sample of this sample type of the database (repeatable)")
    parser.add_argument("--nick", action="append", default=[], help="measure this sample (repeatable)")
    parser.add_argument("--era", help="only the samples of this era among those of --sample-type")
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--json", type=Path, help="write the measurement here")
    args = parser.parse_args()
    if args.table:
        print(module([json.loads(path.read_text()) for path in args.table], [path.name for path in args.table]), end="")
        return
    samples = chosen(json.loads(args.database.read_text()), args.sample_type, args.nick, args.era)
    if not samples:
        raise SystemExit("no sample: give --sample-type or --nick")
    result = measure(samples)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1))
    for entry in result.values():
        print(f'{entry["nick"]}: {json.dumps({name: round(entry[block][key], DIGITS) for name, (block, key) in NORMS.items()})}')


if __name__ == "__main__":
    main()
