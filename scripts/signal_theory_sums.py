"""Measure the inclusive means of the signal theory weights (NN spec SH14, systematics.SIGNAL_THEORY_NORM) of one era
from the NanoAOD of the powheg kappa_lambda = 1 sample, which the sample database names (its `dbs`):

    source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh      # uproot, XRootD
    export X509_USER_PROXY=/tmp/x509up_u12108
    python scripts/signal_theory_sums.py --era 2018 [--pdf-files 1]

The QCD-scale and parton-shower means are exact: the Runs trees of every file (LHEScaleSumw, PSSumw, each the sum of
genWeight times the weight over genEventSumw), combined with genEventSumw. The PDF means are those of the per-event
weights 1 +- sqrt(sum_i (w_i - 1)^2) over the Hessian members 1-100 of LHEPdfWeight (CROWN event::reweighting::LHEpdf),
which the Runs tree cannot give: from the Events of the first --pdf-files files, signed like the analysis (the powheg
weights have one magnitude, so sign(genWeight) and genWeight weight alike). It prints the entry of the era.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import uproot

from bbtautau_shapesmith.systematics import SIGNAL_THEORY_INDICES

REPO = Path(__file__).resolve().parents[1]
DATABASE = REPO.parent / "KingMaker_sample_database" / "nanoAOD_v15" / "datasets.json"
SIGNAL = "GluGluToHHTo2B2T_kl-1p00_kt-1p00_c2-0p00"
REDIRECTOR = "root://cms-xrd-global.cern.ch/"
DASGOCLIENT = "/cvmfs/cms.cern.ch/common/dasgoclient"
PDF_MEMBERS = slice(1, 101)  # the 100 Hessian members of NNPDF31_nnlo_as_0118_mc_hessian_pdfas (LHA IDs 325300-325402)


def dataset(era: str, database: Path) -> str:
    """The DAS name of the era's powheg kappa_lambda = 1 sample."""
    (entry,) = [e for nick, e in json.loads(database.read_text()).items() if nick.startswith(SIGNAL) and e["era"] == era]
    return entry["dbs"]


def files(name: str) -> list[str]:
    output = subprocess.run([DASGOCLIENT, f"-query=file dataset={name}"], check=True, capture_output=True, text=True).stdout
    return sorted(line for line in output.split() if line.endswith(".root"))


def runs_means(urls: list[str]) -> tuple[dict[str, np.ndarray], float, float]:
    """The genEventSumw-weighted means of LHEScaleSumw and PSSumw over every Runs entry, genEventSumw and the ratio
    (sum w)^2 / (N sum w^2), which is the squared fraction (N+ - N-) / N for weights of one magnitude."""
    sums, total, count, total_squared = {}, 0.0, 0.0, 0.0
    for url in urls:
        with uproot.open(url, timeout=600) as handle:
            runs = handle["Runs"].arrays(["genEventSumw", "genEventCount", "genEventSumw2", "LHEScaleSumw", "PSSumw"], library="np")
        for entry in range(len(runs["genEventSumw"])):
            weight = float(runs["genEventSumw"][entry])
            total, count, total_squared = total + weight, count + float(runs["genEventCount"][entry]), total_squared + float(runs["genEventSumw2"][entry])
            for key in ("LHEScaleSumw", "PSSumw"):
                sums[key] = sums.get(key, 0.0) + weight * np.asarray(runs[key][entry], dtype=float)
    return {key: value / total for key, value in sums.items()}, total, total**2 / (count * total_squared)


def pdf_means(urls: list[str]) -> tuple[float, float, int]:
    """The signed means of the per-event PDF weights 1 + q and 1 - q, q = sqrt(sum_i (w_i - 1)^2), and the event count."""
    up = down = norm = 0.0
    n = 0
    for url in urls:
        with uproot.open(url, timeout=600) as handle:
            for arrays in handle["Events"].iterate(["genWeight", "LHEPdfWeight"], step_size="200 MB", library="np"):
                sign = np.sign(arrays["genWeight"])
                q = np.array([np.sqrt(np.sum((np.asarray(w, dtype=float)[PDF_MEMBERS] - 1.0) ** 2)) for w in arrays["LHEPdfWeight"]])
                up, down, norm, n = up + np.sum(sign * (1 + q)), down + np.sum(sign * (1 - q)), norm + np.sum(sign), n + len(q)
    return up / norm, down / norm, n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--era", required=True)
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--pdf-files", type=int, default=1, help="files whose Events give the PDF means (0: none)")
    args = parser.parse_args()
    name = dataset(args.era, args.database)
    urls = [REDIRECTOR + path for path in files(name)]
    print(f"{name}: {len(urls)} files")
    means, total, sign_fraction = runs_means(urls)
    print(f"genEventSumw {total:.6g}, (sum w)^2 / (N sum w^2) = {sign_fraction:.6f}, i.e. (N+ - N-) / N = {np.sqrt(sign_fraction):.6f}")
    entry = {name: round(float(means[key][index]), 5) for name, (key, index) in SIGNAL_THEORY_INDICES.items() if key != "pdf"}
    if args.pdf_files:
        up, down, n = pdf_means(urls[: args.pdf_files])
        print(f"PDF from {n} events of {args.pdf_files} file(s)")
        entry.update({f"pdf{direction}": round(float(value), 5) for direction, value in (("Up", up), ("Down", down))})
    print(f'    "{args.era}": {json.dumps(entry)},')


if __name__ == "__main__":
    main()
