"""Assemble the HH->bbtautau Analysis for ShapeSmith from the run configuration switches; ML export configuration."""
from __future__ import annotations

from shapesmith.config import RunConfig
from shapesmith.model import Analysis, AnalysisError, Estimator, MLExportConfig, Process

from bbtautau_shapesmith.constants import ERA, LUMI_PB
from bbtautau_shapesmith.processes import SIGNAL, categories, control_variables, tau_processes
from bbtautau_shapesmith.samples import DEFAULT_SAMPLE_LIST, sample_database, samples
from bbtautau_shapesmith.selection import channel_definition
from bbtautau_shapesmith.systematics import btag_variations, lnn, style

ML_VARIABLES = (
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
    return MLExportConfig(variables=ML_VARIABLES, processes=tuple(labels), label_of=labels, region_of=region_of)


DEFAULT_SWITCHES = {"jet_fakes": "mc", "embedding": False, "nn_friend": False, "sample_list": DEFAULT_SAMPLE_LIST, "control_regions": False}


def build(config: RunConfig) -> Analysis:
    if "production" in config.switches:
        raise AnalysisError("switch production was renamed to sample_list; use ntuples.base to select production output")
    unknown = set(config.switches) - set(DEFAULT_SWITCHES)
    if unknown:
        raise AnalysisError(f"unknown analysis switches: {sorted(unknown)}")
    if not isinstance(config.switches.get("control_regions", False), bool):
        raise AnalysisError("control_regions must be a boolean")
    switches = {**DEFAULT_SWITCHES, **config.switches}
    jet_fakes, embedding, nn_friend = switches["jet_fakes"], bool(switches["embedding"]), bool(switches["nn_friend"])
    if switches["control_regions"] and jet_fakes != "mc":
        raise AnalysisError("control_regions requires jet_fakes: mc for raw pass/fail data/MC comparisons")
    table = tau_processes(jet_fakes, embedding)
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
        channels={channel: channel_definition(channel, jet_fakes, switches["control_regions"]) for channel in config.channels},
        samples=samples(sample_database(config), switches["sample_list"]),
        processes=table,
        signal=SIGNAL,
        categories=categories() if nn_friend else (),
        control_variables=control_variables(nn_friend),
        weight_variations=btag_variations(),
        lnn=lnn(table, estimator.output),
        estimator=estimator,
        style=style(),
        ml=ml_config(table, jet_fakes),
    )
