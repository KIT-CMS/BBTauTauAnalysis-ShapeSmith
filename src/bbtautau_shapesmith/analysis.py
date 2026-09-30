"""Assemble the HH->bbtautau Analysis of the tau channels from the run configuration switches; ML export configuration."""
from __future__ import annotations

from shapesmith.config import RunConfig
from shapesmith.model import ABCD, Analysis, Channel, DataMinus, MLExportConfig, Process, Sample, TemplateShift

from bbtautau_shapesmith import cuts
from bbtautau_shapesmith.constants import ERA, LUMI_PB
from bbtautau_shapesmith.processes import EMBEDDED, GENUINE, JET_FAKE, LEPTON_FAKE, SIGNAL, SPLITS, backgrounds_in, tau_processes
from bbtautau_shapesmith.samples import sample_database, samples
from bbtautau_shapesmith.style import style
from bbtautau_shapesmith.switches import TauSwitches, parse
from bbtautau_shapesmith.systematics import btag_variations, embedding_variations, ff_variations, lnn
from bbtautau_shapesmith.variables import categories, control_variables

ML_VARIABLES = (
    "pt_1", "pt_2", "m_vis", "pt_vis", "mt_tot", "pt_tautau", "deltaR_ditaupair", "met", "n_jets", "n_bjets",
    "jpt_1", "jpt_2", "mjj", "pt_dijet", "bpair_pt_1", "bpair_pt_2", "bpair_btag_value_1", "bpair_btag_value_2",
    "bpair_m_inv", "bpair_pt_dijet", "bpair_deltaR", "pt_tautaubb", "mass_tautaubb",
)
# NN training labels; the embedded genuine di-tau events take the place of ZTT in the DY class
LABEL_OF_GROUP = {"HH2B2Tau": "is_HH2B2Tau", "DY": "is_DY", EMBEDDED: "is_DY", "TT": "is_TT", "ST": "is_ST", "VV": "is_VV"}


def label_of(process: Process) -> str:
    if process.name in JET_FAKE:
        return "is_jetFakes"
    return LABEL_OF_GROUP.get(process.group, "is_Other")


def ml_config(processes: tuple[Process, ...], jet_fakes: str) -> MLExportConfig:
    labels = {p.name: label_of(p) for p in processes if p.role in ("signal", "background")}
    region_of = {}
    if jet_fakes == "ff":
        labels["jetFakes"] = "is_jetFakes"
        region_of["jetFakes"] = "anti_iso"
    return MLExportConfig(variables=ML_VARIABLES, processes=tuple(labels), label_of=labels, region_of=region_of)


def estimators(processes: tuple[Process, ...], switches: TauSwitches) -> tuple:
    """jetFakes = data - (genuine + lepton fakes) in the fake-factor region, or QCD from ABCD with every MC subtracted;
    with embedding the ttbar contamination of the embedded sample (+-10 % of genuine ttbar)."""
    genuine_and_lepton_fakes = backgrounds_in(processes, GENUINE) + backgrounds_in(processes, LEPTON_FAKE)
    if switches.jet_fakes == "ff":
        result = (DataMinus("jetFakes", "anti_iso", genuine_and_lepton_fakes),)
    else:
        result = (ABCD("QCD", "abcd_anti_iso", "abcd_same_sign", "abcd_same_sign_anti_iso", genuine_and_lepton_fakes + backgrounds_in(processes, JET_FAKE)),)
    if switches.embedding:
        result += (TemplateShift(f"CMS_htt_emb_ttbar_{ERA}", EMBEDDED, SPLITS["TT"]["T"], 0.1),)
    return result


def column_variations(name: str, switches: TauSwitches) -> tuple:
    if not switches.shape_systematics:
        return ()
    return (embedding_variations(name) if switches.embedding else ()) + (ff_variations(name) if switches.jet_fakes == "ff" else ())


def channel(name: str, switches: TauSwitches, channel_samples: tuple[Sample, ...]) -> Channel:
    table = tau_processes(name, switches.jet_fakes, switches.embedding)
    groups = {p.group for p in table}
    return Channel(
        name=name,
        samples=tuple(s for s in channel_samples if s.group in groups),
        skim=cuts.skim_cuts(name, switches.control_regions),
        cuts=cuts.baseline_cuts(name),
        processes=table,
        regions=cuts.regions(name, switches.jet_fakes) + (cuts.diagnostic_regions(name) if switches.control_regions else ()),
        categories=categories() if switches.nn_friend else (),
        variables=control_variables(switches.nn_friend),
        variations=btag_variations() + column_variations(name, switches),
        estimators=estimators(table, switches),
        keep_columns=("event", "run", "lumi"),
    )


def build(config: RunConfig) -> Analysis:
    switches = parse(TauSwitches, config.switches)
    by_channel = samples(sample_database(config), switches.sample_lists, config.channels)
    channels = {name: channel(name, switches, by_channel[name]) for name in config.channels}
    first = next(iter(channels.values()))  # every tau channel has the same processes and estimators
    return Analysis(
        name="hh_bbtautau_2018_v15",
        era=ERA,
        lumi_pb=LUMI_PB,
        signal=SIGNAL,
        channels=channels,
        lnn=lnn(first.processes, first.estimators[0].output),
        style=style(),
        ml=ml_config(first.processes, switches.jet_fakes),
    )
