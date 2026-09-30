"""Sample table: the nicks of the production's sample lists, their group and channels by nick, the normalisation
from the KingMaker database.

`inventory/<name>.txt` is an unchanged copy of a KingMaker sample list: one nick per line, looked up by nick in the
sample database (kind from its sample_type, cross section, event count, generator weight).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from shapesmith.config import RunConfig
from shapesmith.model import AnalysisError, Sample
from shapesmith.samples import normalisation, read_sample_list

INVENTORY_DIR = Path(__file__).resolve().parents[2] / "inventory"


def inventory_path(sample_list: str) -> Path:
    """`inventory/<sample_list>.txt`, the verbatim KingMaker sample list."""
    path = INVENTORY_DIR / f"{sample_list}.txt"
    if not path.exists():
        known = sorted(p.stem for p in INVENTORY_DIR.glob("*.txt"))
        raise FileNotFoundError(f"no inventory for sample list {sample_list!r}; known: {known}")
    return path


def sample_database(config: RunConfig) -> Path:
    """The KingMaker datasets.json named in the run configuration (cross sections, event counts, generator weights)."""
    if config.sample_database is None:
        raise AnalysisError("sample_database (path to a KingMaker datasets.json) is missing in the run configuration")
    return config.sample_database


# Embedded samples (EmbeddingRun<era>...): the final-state token in the nick decides group and channel.
EMBEDDING_PREFIX = "EmbeddingRun"
EMBEDDING_ROUTES = {"_mutau_": ("EMB", ("mt",)), "_eltau_": ("EMB", ("et",)), "_tautau_": ("EMB", ("tt",)), "_muemb_": ("MUEMB", ("mm",))}

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


def route(nick: str) -> tuple[str, tuple[str, ...] | None]:
    """Group and channels (None: every channel) of a nick."""
    if nick.startswith(EMBEDDING_PREFIX):
        for token, group_and_channels in EMBEDDING_ROUTES.items():
            if token in nick:
                return group_and_channels
        raise KeyError(f"embedded sample {nick} has none of the final states {', '.join(EMBEDDING_ROUTES)}")
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


def samples(database: Path, sample_lists: list[str], channels: list[str]) -> dict[str, tuple[Sample, ...]]:
    """The samples of every channel, in sample-list order; a nick listed twice raises."""
    nicks = [nick for name in sample_lists for nick in read_sample_list(inventory_path(name))]
    repeated = sorted(nick for nick, count in Counter(nicks).items() if count > 1)
    if repeated:
        raise AnalysisError(f"nicks in more than one of the sample lists {sample_lists}:\n" + "\n".join(repeated))
    values = normalisation(database, nicks)
    result = {channel: [] for channel in channels}
    for nick in nicks:
        group, routed = route(nick)
        row = values[nick]
        sample = Sample(nick, group, row["kind"], float(row["xsec"]), int(row["nevents"]), float(row["generator_weight"]), cut=cut_of(nick))
        for channel in channels:
            if routed is None or channel in routed:
                result[channel].append(sample)
    return {channel: tuple(chosen) for channel, chosen in result.items()}
