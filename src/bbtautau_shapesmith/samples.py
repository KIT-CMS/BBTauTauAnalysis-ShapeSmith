"""Sample table: the nicks of the production's sample lists, their group and channels by nick, the normalisation
from the KingMaker database.

`inventory/<name>.txt` is an unchanged copy of a KingMaker sample list: one nick per line, looked up by nick in the
sample database (kind from its sample_type, cross section, event count, generator weight). Every nick must be a sample
of the run's era (its database era).

A borrowed sample (v7 spec 3.3, 6.3) is the line `<source era>/<nick>` of an inventory: its n-tuples are those of the
source era, it is normalised with the luminosity of the run. Only the borrowings of BORROWED are accepted: VH -> tautau
has no UL16 (postVFP) NanoAODv15 sample, so the 2016postVFP run takes the 2016preVFP ones; the datacards give the
process an extra lnN (systematics.BORROWED_LNN).

A run of a combined era (constants.COMBINED_ERAS, the 2016 fake-factor measurement, E18) takes the samples of every
member era: each sample reads the n-tuples of its database era and is normalised to its luminosity
(Sample.lumi_scale = its era's share of the combined luminosity). It borrows nothing. A sample list named with the
combined era (`sm_2016_binned`, from `sm_{era}_binned` of a run YAML) stands for the lists of its member eras
(`sm_2016preVFP_binned`, `sm_2016postVFP_binned`) when no inventory of that name exists.

The v7 inventories are named as the KingMaker lists (`sm_<era>_binned`, `_binned_mc`, `_embedding`,
`_tau_id_measurement`); `sm_<era>_borrowed` holds the borrowed lines of an era (empty where it borrows nothing), so a
run YAML lists `sm_{era}_borrowed` in every era.
"""
from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from pathlib import Path

from shapesmith.config import RunConfig
from shapesmith.model import AnalysisError, Sample
from shapesmith.samples import normalisation, read_sample_list

from bbtautau_shapesmith.constants import COMBINED_ERAS, ERAS, LUMI_PB, lumi_pb, member_eras
from bbtautau_shapesmith.processes import NODE_SM, SIGNAL, VH_TAUTAU

INVENTORY_DIR = Path(__file__).resolve().parents[2] / "inventory"

logger = logging.getLogger(__name__)


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
    ("SingleElectron_", "data", ("et", "ee")),  # 2016, 2017
    ("EGamma_", "data", ("et", "ee")),  # 2018
    ("Tau_", "data", ("tt",)),
    ("GluGluToHHTo2B2T_kl-1p00", SIGNAL, None),  # powheg NLO, kappa_lambda = 1: the signal (E23)
    ("GluGluToHHTo2B2T_kl-", "HH2B2Tau_kl", None),  # the other kappa_lambda points: no process (for later scans)
    ("GluGluToHHTo2B2Tau", NODE_SM, None),  # madgraph node_SM: the cross-check, or the signal of a list without powheg
    ("GluGluHToTauTau", "ggH", None),
    ("VBFH", "qqH", None),
    ("ttH", "ttH", None),
    ("ZHToTauTau", VH_TAUTAU, None),
    ("WplusHToTauTau", VH_TAUTAU, None),
    ("WminusHToTauTau", VH_TAUTAU, None),
    ("WminusH", "VH", None),
    ("WplusH", "VH", None),
    ("ZH_", "VH", None),
    ("ggZH", "VH", None),
    ("DYJetsToLL", "DY", None),
    ("WJetsToLNu", "W", None),
    ("EWK", "EWK", None),
    ("ST_", "ST", None),
    ("TTTo", "TT", None),
    ("TTW", "TTV", None),  # TTWJetsToLNu, TTWJetsToQQ, TTWW, TTWZ, TTWH
    ("TTZ", "TTV", None),
    ("ttW", "TTV", None),
    ("ttZ", "TTV", None),
    ("WW", "VV", None),
    ("WZ", "VV", None),  # WZTo4Q included
    ("ZZ", "VV", None),  # ZZTo2Nu2Q_5f included
)
# group whose samples stand in for the signal when the sample lists have none of it: the 2018 productions up to v6 had
# the madgraph node_SM sample as their signal
STAND_IN = {NODE_SM: SIGNAL}
# era of the run -> group -> the era whose samples it borrows (3.3)
BORROWED = {"2016postVFP": {VH_TAUTAU: "2016preVFP"}}


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


# (nick prefix, cut applied at skim time, nick prefix of the partition that needs it); first match wins, normalisation
# stays that of the whole sample. The DY LHEFilterPtZ bins lack every zero-parton event: when they are listed, that
# part comes from the inclusive sample and the bins cover npartons >= 1; without them the inclusive sample is uncut.
SAMPLE_CUTS = (
    ("DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX", "npartons == 0", "DYJetsToLL_LHEFilterPtZ-"),
)


def cut_of(nick: str, nicks: list[str]) -> str | None:
    """The skim-time cut of a sample, given all nicks of the sample lists."""
    for prefix, cut, partition in SAMPLE_CUTS:
        if nick.startswith(prefix):
            return cut if any(other.startswith(partition) for other in nicks) else None
    return None


def inventory_names(sample_list: str, era: str) -> tuple[str, ...]:
    """The inventories of a sample list in a run of `era`: the list itself, or, in a combined era without an inventory
    of that name, the lists of its member eras with the era token replaced (`sm_2016_binned` -> `sm_2016preVFP_binned`,
    `sm_2016postVFP_binned`)."""
    tokens = sample_list.split("_")
    if era in COMBINED_ERAS and era in tokens and not (INVENTORY_DIR / f"{sample_list}.txt").exists():
        return tuple("_".join(member if token == era else token for token in tokens) for member in member_eras(era))
    return (sample_list,)


def parse_entry(line: str) -> tuple[str, str | None]:
    """An inventory line: (nick, None), or (nick, source era) for a borrowed sample `<source era>/<nick>`."""
    source, separator, nick = line.partition("/")
    return (nick, source) if separator else (line, None)


def inventory_nicks(sample_list: str) -> tuple[str, ...]:
    """The nicks of an inventory, borrowed ones without their source era."""
    return tuple(parse_entry(line)[0] for line in read_sample_list(inventory_path(sample_list)))


def _era_problems(nick: str, group: str, source: str | None, database_era: str | None, era: str) -> list[str]:
    """A nick must be a sample of the run's era (of a member era in a combined era), a borrowed one a sample of its
    source era that BORROWED allows."""
    if era in COMBINED_ERAS:
        if source is not None:
            return [f"{source}/{nick}: a run of the combined era {era} borrows nothing"]
        return [] if database_era in member_eras(era) else [f"{nick}: a {database_era} sample in a {era} ({' + '.join(member_eras(era))}) run"]
    if source is None:
        return [] if database_era in (None, era) else [f"{nick}: a {database_era} sample in a {era} run"]
    if source not in ERAS or BORROWED.get(era, {}).get(group) != source:
        allowed = ", ".join(f"{g} from {e}" for g, e in BORROWED.get(era, {}).items()) or "none"
        return [f"{source}/{nick}: the {era} run does not borrow {group} from {source} (allowed: {allowed})"]
    return [] if database_era in (None, source) else [f"{source}/{nick}: the database has it as a {database_era} sample"]


def _stand_in(chosen: list[Sample]) -> list[Sample]:
    """The samples of a STAND_IN group take the place of the group they stand in for if the lists have none of it."""
    groups = {s.group for s in chosen}
    replaced = {group: target for group, target in STAND_IN.items() if group in groups and target not in groups}
    for group, target in replaced.items():
        logger.info(f"no {target} sample listed: the {group} samples are the {target} samples")
    return [dataclasses.replace(s, group=replaced[s.group]) if s.group in replaced else s for s in chosen]


def samples(database: Path, sample_lists: list[str], channels: list[str], era: str) -> dict[str, tuple[Sample, ...]]:
    """The samples of every channel of a run in `era`, in sample-list order; a nick listed twice, a nick of another era
    and a borrowing BORROWED does not allow raise."""
    names = [member for name in sample_lists for member in inventory_names(name, era)]
    if names != list(sample_lists):
        logger.info(f"sample lists of the combined era {era}: {', '.join(names)}")
    entries = [parse_entry(line) for name in names for line in read_sample_list(inventory_path(name))]
    nicks = [nick for nick, _ in entries]
    repeated = sorted(nick for nick, count in Counter(nicks).items() if count > 1)
    if repeated:
        raise AnalysisError(f"nicks in more than one of the sample lists {names}:\n" + "\n".join(repeated))
    logger.info(f"{len(nicks)} nicks from the sample lists {', '.join(names)}, normalisation from {database}")
    values = normalisation(database, nicks)
    problems, chosen = [], []
    for nick, source in entries:
        group, routed = route(nick)
        row = values[nick]
        problems += _era_problems(nick, group, source, row.get("era"), era)
        own_era, lumi_scale = source, 1.0
        if era in COMBINED_ERAS and row.get("era") in member_eras(era):
            own_era, lumi_scale = row["era"], (LUMI_PB[row["era"]] / lumi_pb(era) if row["kind"] == "mc" else 1.0)
        sample = Sample(nick, group, row["kind"], float(row["xsec"]), int(row["nevents"]), float(row["generator_weight"]), cut=cut_of(nick, nicks), era=own_era, lumi_scale=lumi_scale)
        logger.debug(f"{nick}: group {group}, {row['kind']}, channels {', '.join(routed) if routed else 'all'}, xsec {sample.xsec:g} pb, {sample.nevents} events, generator weight {sample.generator_weight:g}")
        if sample.cut is not None:
            logger.info(f"{nick}: skim-time cut {sample.cut}")
        if source is not None:
            logger.info(f"{nick}: borrowed from {source}, normalised with the {era} luminosity")
        chosen.append((sample, routed))
    if problems:
        raise AnalysisError(f"sample lists {', '.join(names)} of the {era} run:\n" + "\n".join(problems))
    standing_in = _stand_in([sample for sample, _ in chosen])
    result = {channel: [] for channel in channels}
    for sample, (_, routed) in zip(standing_in, chosen):
        for channel in channels:
            if routed is None or channel in routed:
                result[channel].append(sample)
    return {channel: tuple(listed) for channel, listed in result.items()}
