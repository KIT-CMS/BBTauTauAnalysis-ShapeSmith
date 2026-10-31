"""The v7 inventories of the Run-2 tests: the sample lists of every era (copies of the CROWN lists) and the nicks in
them.

sm_<era>_binned (data and MC), sm_<era>_embedding, sm_<era>_tau_id_measurement and sm_<era>_borrowed (2016postVFP:
VH -> tautau of 2016preVFP, spec 3.3; empty in the other eras) are in inventory/, their database entries in
tests/fixtures/datasets.json.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from shapesmith.samples import read_sample_list

from tests.helpers import DATABASE, ERAS, INVENTORY

KINDS = {"binned": "binned", "binned_mc": "binned_mc", "embedding": "embedding", "tau_id": "tau_id_measurement", "borrowed": "borrowed"}
VH_TAUTAU = ("ZHToTauTau_M125", "WplusHToTauTau_M125", "WminusHToTauTau_M125")  # nick prefixes
POWHEG = {"1p00": "GluGluToHHTo2B2T_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13TeV-powheg-pythia8", "0p00": "GluGluToHHTo2B2T_kl-0p00_kt-1p00_c2-0p00_TuneCP5_13TeV-powheg-pythia8"}
NODE_SM_PREFIX = "GluGluToHHTo2B2Tau_TuneCP5_PSWeights_node_SM"

__all__ = ["DATABASE", "ERAS", "KINDS", "POWHEG", "VH_TAUTAU", "install", "list_name", "nick", "sample_lists"]


def list_name(kind: str, era: str) -> str:
    """The inventory of a kind (binned, binned_mc, embedding, tau_id, borrowed) in an era."""
    return f"sm_{era}_{KINDS[kind]}"


def sample_lists(era: str, *kinds: str) -> list[str]:
    """The inventories of an era; with binned also its borrowed list (empty but in 2016postVFP)."""
    names = [list_name(kind, era) for kind in kinds]
    return names + [list_name("borrowed", era)] if "binned" in kinds else names


def nick(prefix: str, era: str, kind: str = "binned") -> str:
    """The one nick of an era's inventory that starts with `prefix`."""
    (found,) = [n for n in read_sample_list(INVENTORY / f"{list_name(kind, era)}.txt") if n.startswith(prefix)]
    return found


def install(directory: Path) -> Path:
    """Copies of every inventory in `directory` (for tests that add a list there); returns the fixture database."""
    directory.mkdir(parents=True, exist_ok=True)
    for inventory in INVENTORY.glob("*.txt"):
        shutil.copyfile(inventory, directory / inventory.name)
    return DATABASE
