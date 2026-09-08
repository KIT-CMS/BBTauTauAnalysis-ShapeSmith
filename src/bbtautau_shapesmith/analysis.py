"""Assemble the HH->bbtautau Analysis for ShapeSmith from the run configuration switches; ML export configuration."""
from __future__ import annotations

from pathlib import Path

from shapesmith.config import RunConfig
from shapesmith.model import Analysis, AnalysisError, Estimator, MLExportConfig, Process

from bbtautau_shapesmith.constants import ERA, LUMI_PB
from bbtautau_shapesmith.processes import SIGNAL, categories, control_variables, processes
from bbtautau_shapesmith.samples import samples
from bbtautau_shapesmith.selection import channel_definition
from bbtautau_shapesmith.systematics import lnn, style, weight_variations

VARIABLES = (
    "pt_1", "pt_2", "m_vis", "pt_vis", "mt_tot", "pt_tautau", "deltaR_ditaupair", "met", "n_jets", "n_bjets",
    "jpt_1", "jpt_2", "mjj", "pt_dijet", "bpair_pt_1", "bpair_pt_2", "bpair_btag_value_1", "bpair_btag_value_2",
    "bpair_m_inv", "bpair_pt_dijet", "bpair_deltaR", "pt_tautaubb", "mass_tautaubb",
)
LABEL_OF_GROUP = {"HH2B2Tau": "is_HH2B2Tau", "DY": "is_DY", "TT": "is_TT", "ST": "is_ST", "VV": "is_VV"}


def label_of(process: Process) -> str:
    if process.kind == "jet_fake":
        return "is_jetFakes"
    return LABEL_OF_GROUP.get(process.group, "is_Other")


def ml_config(processes: tuple[Process, ...], jet_fakes: str) -> MLExportConfig:
    exported = [p for p in processes if p.kind != "data"]
    labels = {p.name: label_of(p) for p in exported}
    region_of = {}
    if jet_fakes == "ff":
        labels["jetFakes"] = "is_jetFakes"
        region_of["jetFakes"] = "anti_iso"
    return MLExportConfig(variables=VARIABLES, processes=tuple(labels), label_of=labels, region_of=region_of)


DEFAULT_SWITCHES = {"jet_fakes": "mc", "embedding": False, "nn_friend": False, "sample_list": "sm2018_binned_v2"}


def _database(config: RunConfig) -> Path:
    if config.sample_database is None:
        raise AnalysisError("sample_database (path to a KingMaker datasets.json) is missing in the run configuration")
    return config.sample_database


def build(config: RunConfig) -> Analysis:
    switches = {**DEFAULT_SWITCHES, **config.switches}
    jet_fakes, embedding, nn_friend = switches["jet_fakes"], bool(switches["embedding"]), bool(switches["nn_friend"])
    table = processes(jet_fakes, embedding)
    genuine = ("EMB",) if embedding else tuple(p.name for p in table if p.kind == "true_tau")
    lepton_fakes = tuple(p.name for p in table if p.kind == "lepton_fake")
    if jet_fakes == "ff":
        estimator = Estimator("fake_factors", {"anti_iso": "anti_iso"}, genuine + lepton_fakes, "jetFakes")
    else:
        jet_fake_mc = tuple(p.name for p in table if p.kind == "jet_fake")
        estimator = Estimator("abcd", {"B": "abcd_anti_iso", "C": "abcd_same_sign", "D": "abcd_same_sign_anti_iso"}, genuine + lepton_fakes + jet_fake_mc, "QCD")
    return Analysis(
        name="hh_bbtautau_2018_v15",
        era=ERA,
        lumi_pb=LUMI_PB,
        channels={channel: channel_definition(channel, jet_fakes) for channel in config.channels},
        samples=samples(_database(config), switches["sample_list"]),
        processes=table,
        signal=SIGNAL,
        categories=categories() if nn_friend else (),
        control_variables=control_variables(nn_friend),
        weight_variations=weight_variations(),
        lnn=lnn(table, estimator.output),
        estimator=estimator,
        style=style(),
        ml=ml_config(table, jet_fakes),
    )
