"""Sample table: group and channel assignment by nick prefix, normalisation from the KingMaker database.

The database renames nicks now and then (the 2018 v15 nicks gained a campaign suffix in 2026-08) while the
CROWN output keeps the production-time nick, so the inventory carries the DBS path of every sample and
shapesmith resolves the database entry by nick first and by DBS path second.
"""
from __future__ import annotations

from pathlib import Path

from shapesmith.config import RunConfig
from shapesmith.model import AnalysisError, Sample
from shapesmith.samples import normalisation, read_inventory

INVENTORY_DIR = Path(__file__).resolve().parents[2] / "inventory"
DEFAULT_SAMPLE_LIST = "sm2018_binned_v2"


def inventory_path(sample_list: str = DEFAULT_SAMPLE_LIST) -> Path:
    """`inventory/<sample_list>.txt`: one `<nick> <dbs>` line per sample to process."""
    path = INVENTORY_DIR / f"{sample_list}.txt"
    if not path.exists():
        known = sorted(p.stem for p in INVENTORY_DIR.glob("*.txt"))
        raise FileNotFoundError(f"no inventory for sample_list {sample_list!r}; known: {known}")
    return path


def sample_database(config: RunConfig) -> Path:
    """The KingMaker datasets.json named in the run configuration (cross sections, event counts, generator weights)."""
    if config.sample_database is None:
        raise AnalysisError("sample_database (path to a KingMaker datasets.json) is missing in the run configuration")
    return config.sample_database


# (nick prefix, group, channels); first match wins, so put the more specific prefixes first.
# CROWN writes every data stream into every scope, so a stream is routed to the channels whose trigger it carries.
GROUP_RULES = (
    ("SingleMuon_", "data", ("mt", "em", "mm")),
    ("EGamma_", "data", ("et", "ee")),
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


def group_of(nick: str) -> tuple[str, tuple[str, ...] | None]:
    for prefix, group, channels in GROUP_RULES:
        if nick.startswith(prefix):
            return group, channels
    raise KeyError(f"no group rule for sample {nick}")


# (nick prefix, cut applied at skim time); first match wins, normalisation stays that of the whole sample.
# The DY LHEFilterPtZ bins lack every zero-parton event: that part comes from the inclusive sample, the bins cover npartons >= 1.
SAMPLE_CUTS = (
    ("DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX", "npartons == 0"),
)


def cut_of(nick: str) -> str | None:
    for prefix, cut in SAMPLE_CUTS:
        if nick.startswith(prefix):
            return cut
    return None


def samples(database: Path, sample_list: str = DEFAULT_SAMPLE_LIST) -> tuple[Sample, ...]:
    entries = read_inventory(inventory_path(sample_list))
    values = normalisation(database, entries)
    result = []
    for entry in entries:
        group, channels = group_of(entry.nick)
        row = values[entry.nick]
        result.append(Sample(entry.nick, group, row["kind"], float(row["xsec"]), int(row["nevents"]), float(row["generator_weight"]), channels, cut=cut_of(entry.nick)))
    return tuple(result)
