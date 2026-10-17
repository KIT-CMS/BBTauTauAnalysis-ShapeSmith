"""Per-era fixtures of the Run-2 tests: sample lists and a sample database of all four eras.

The lists of an era are the 2018 fixture lists with the era in place of 2018 in every nick (EGamma becomes SingleElectron
before 2018), plus the v7 additions of spec 3.1: the powheg signal (kappa_lambda 1 and 0) and VH -> tautau; 2016postVFP
borrows VH -> tautau from 2016preVFP (3.3). They stand in for the inventories (copies of the KingMaker lists of v7)
until those exist; the database entries keep the 2018 normalisation with the era of the copy.
"""
from __future__ import annotations

import json
from pathlib import Path

from shapesmith.samples import read_sample_list

from tests.helpers import DATABASE, EMBEDDING_LIST, INVENTORY

ERAS = ("2016preVFP", "2016postVFP", "2017", "2018")
CAMPAIGN = {"2016preVFP": "UL16APV", "2016postVFP": "UL16", "2017": "UL17", "2018": "UL18"}
BASE_LISTS = {"binned": "sm2018_binned_v6", "embedding": EMBEDDING_LIST, "tau_id": "sm2018_tau_id_measurement"}
VH_TAUTAU = ("ZHToTauTau_M125_CP5_13TeV-powheg-pythia8", "WplusHToTauTau_M125_CP5_13TeV-powheg-pythia8", "WminusHToTauTau_M125_CP5_13TeV-powheg-pythia8")
POWHEG = {kl: f"GluGluToHHTo2B2T_kl-{kl}_kt-1p00_c2-0p00_TuneCP5_13TeV-powheg-pythia8" for kl in ("1p00", "0p00")}


def era_nick(nick: str, era: str) -> str:
    """The 2018 nick as a nick of the era."""
    if era == "2018":
        return nick
    return nick.replace("EGamma_", "SingleElectron_").replace("UL18", CAMPAIGN[era]).replace("2018", era)


def mc_nick(primary: str, era: str) -> str:
    return f"{primary}_RunIISummer20{CAMPAIGN[era]}NanoAODv15-150X_mc{era}_realistic_v1-v1"


def list_name(kind: str, era: str) -> str:
    return f"run2_{era}_{kind}"


def write(directory: Path) -> Path:
    """Write the lists (run2_<era>_binned, _embedding, _tau_id, run2_2016postVFP_borrowed) and their database into
    `directory`; returns the database."""
    directory.mkdir(parents=True, exist_ok=True)
    base = json.loads(DATABASE.read_text())
    database = {}

    def add(nick: str, entry: dict, era: str) -> str:
        database[nick] = {**entry, "nick": nick, "era": era}
        return nick

    template = base[next(n for n in base if base[n]["sample_type"] == "hh2b2tau")]
    for era in ERAS:
        for kind, source in BASE_LISTS.items():
            nicks = [add(era_nick(nick, era), base[nick], era) for nick in read_sample_list(INVENTORY / f"{source}.txt")]
            if kind == "binned":
                nicks += [add(mc_nick(primary, era), {**template, "sample_type": "hh2b2tau", "xsec": 0.002221}, era) for primary in POWHEG.values()]
                if era != "2016postVFP":
                    nicks += [add(mc_nick(primary, era), {**template, "sample_type": "rem_htautau", "xsec": 0.05}, era) for primary in VH_TAUTAU]
            (directory / f"{list_name(kind, era)}.txt").write_text("".join(f"{nick}\n" for nick in nicks))
    borrowed = [f"2016preVFP/{mc_nick(primary, '2016preVFP')}" for primary in VH_TAUTAU]
    (directory / "run2_2016postVFP_borrowed.txt").write_text("".join(f"{line}\n" for line in borrowed))
    path = directory / "datasets.json"
    path.write_text(json.dumps(database))
    return path


def sample_lists(era: str, *kinds: str) -> list[str]:
    """The fixture lists of an era (binned and, with embedding, the embedding list; 2016postVFP its borrowed list)."""
    names = [list_name(kind, era) for kind in kinds]
    if era == "2016postVFP" and "binned" in kinds:
        names.append("run2_2016postVFP_borrowed")
    return names
