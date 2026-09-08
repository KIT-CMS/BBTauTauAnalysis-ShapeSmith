"""Sample table: group and channel assignment by nick prefix, normalisation from the KingMaker database.

The database renames nicks now and then (the 2018 v15 nicks gained a campaign suffix in 2026-08) while the
CROWN output keeps the production-time nick, so the inventory carries the DBS path of every sample and
shapesmith resolves the database entry by nick first and by DBS path second.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from shapesmith.model import Sample
from shapesmith.samples import normalisation, read_inventory

INVENTORY_DIR = Path(__file__).resolve().parents[2] / "inventory"
DEFAULT_SAMPLE_LIST = "sm2018_binned_v2"


def inventory_path(sample_list: str = DEFAULT_SAMPLE_LIST) -> Path:
    """`inventory/<sample_list>.txt`: one `<nick> <dbs>` line per sample to process."""
    path = INVENTORY_DIR / f"{sample_list}.txt"
    if not path.exists():
        known = sorted(p.stem for p in INVENTORY_DIR.glob("sm*.txt"))
        raise FileNotFoundError(f"no inventory for sample_list {sample_list!r}; known: {known}")
    return path

# (nick prefix, group, channels); first match wins, so put the more specific prefixes first
GROUP_RULES = (
    ("SingleMuon_", "data", ("mt",)),
    ("EGamma_", "data", ("et",)),
    ("Tau_", "data", ("tt",)),
    ("GluGluToHHTo2B2Tau", "HH2B2Tau", None),
    ("GluGluHToTauTau", "ggH", None),
    ("VBFH", "qqH", None),
    ("ttH", "ttH", None),
    ("WminusH", "VH", None),
    ("WplusH", "VH", None),
    ("ZH_", "VH", None),
    ("ggZH", "VH", None),
    ("DYJetsToLL", "DY", None),
    ("WJetsToLNu", "W", None),
    ("EWK", "EWK", None),
    ("ST_", "ST", None),
    ("TTTo", "TT", None),
    ("TTW", "TTV", None),
    ("TTZ", "TTV", None),
    ("ttW", "TTV", None),
    ("ttZ", "TTV", None),
    ("WW", "VV", None),
    ("WZ", "VV", None),
    ("ZZ", "VV", None),
)

EXPECTED_GROUP_SIZES = Counter({"data": 12, "HH2B2Tau": 1, "DY": 7, "W": 3, "EWK": 7, "ST": 6, "TT": 3, "TTV": 11, "VV": 14, "ggH": 1, "qqH": 2, "ttH": 2, "VH": 10})


def group_of(nick: str) -> tuple[str, tuple[str, ...] | None]:
    for prefix, group, channels in GROUP_RULES:
        if nick.startswith(prefix):
            return group, channels
    raise KeyError(f"no group rule for sample {nick}")


def samples(database: Path, sample_list: str = DEFAULT_SAMPLE_LIST) -> tuple[Sample, ...]:
    entries = read_inventory(inventory_path(sample_list))
    values = normalisation(database, entries)
    result = []
    for entry in entries:
        group, channels = group_of(entry.nick)
        row = values[entry.nick]
        result.append(Sample(entry.nick, group, row["kind"], float(row["xsec"]), int(row["nevents"]), float(row["generator_weight"]), channels))
    return tuple(result)
