"""Assemble the HH->bbtautau Analysis of the tau channels (and the ttbar control channel em) of one era from the run
configuration switches; ML export configuration."""
from __future__ import annotations

from collections.abc import Iterable

from shapesmith.config import RunConfig
from shapesmith.model import ABCD, Analysis, AnalysisError, Category, Channel, DataMinus, MLExportConfig, Process, Sample, TemplateShift

from bbtautau_shapesmith import cuts, dilepton
from bbtautau_shapesmith.constants import BTAG_BINS, CONTROL_CHANNELS, LT_CHANNELS, LUMI_PB, PREFIRING_NUISANCE, TAU_CHANNELS, check_era
from bbtautau_shapesmith.processes import EMBEDDED, GENUINE, JET_FAKE, LEPTON_FAKE, SIGNAL, SPLITS, available, backgrounds_in, tau_processes
from bbtautau_shapesmith.samples import sample_database, samples
from bbtautau_shapesmith.style import style
from bbtautau_shapesmith.switches import TauSwitches, parse
from bbtautau_shapesmith.systematics import (
    btag_variations, embedding_ttbar_nuisance, embedding_variation_sums, embedding_variations, ff_variations, lnn, prefiring_variations,
)
from bbtautau_shapesmith.variables import categories, control_variables, nominal_category

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


def process_union(channels: Iterable[Channel]) -> tuple[Process, ...]:
    """The processes of all channels, each name once in channel order; a process that is auxiliary in one channel and
    a background in another (TTT with and without embedding) takes its background role."""
    union = {}
    for channel in channels:
        for process in channel.processes:
            if process.name not in union or union[process.name].role == "auxiliary":
                union[process.name] = process
    return tuple(union.values())


def ml_config(processes: tuple[Process, ...], jet_fakes: str) -> MLExportConfig:
    labels = {p.name: label_of(p) for p in processes if p.role in ("signal", "background")}
    region_of = {}
    if jet_fakes == "ff":
        labels["jetFakes"] = "is_jetFakes"
        region_of["jetFakes"] = "anti_iso"
    return MLExportConfig(variables=ML_VARIABLES, processes=tuple(labels), label_of=labels, region_of=region_of)


def jet_fakes_output(jet_fakes: str) -> str:
    return "jetFakes" if jet_fakes == "ff" else "QCD"


def diagnostic_estimators(name: str, every_mc: tuple[str, ...]) -> tuple:
    """QCD in the control regions of every b-tag bin: in et and mt the same-sign data minus every MC (OS/SS = 1, which
    only btag0 can measure), also for the lepton anti-isolation region; in tt ABCD with the tau-fail regions."""
    if name not in LT_CHANNELS:
        return tuple(ABCD("QCD", f"{b}_ss_pass", f"{b}_os_fail", f"{b}_ss_fail", every_mc, target=f"{b}_os_pass") for b in BTAG_BINS)
    pairs = [(f"{b}_ss_{tau_id}", f"{b}_os_{tau_id}") for b in BTAG_BINS for tau_id in ("pass", "fail")] + [("lepton_antiiso_ss", "lepton_antiiso")]
    return tuple(DataMinus("QCD", region, every_mc, clip_negative=True, target=target) for region, target in pairs)


def estimators(name: str, era: str, processes: tuple[Process, ...], switches: TauSwitches) -> tuple:
    """jetFakes = data - (genuine + lepton fakes) in the fake-factor region, or QCD from ABCD with every MC subtracted
    (with diagnostics also in the control regions); in an embedded channel the sums of the embedding shifts of DM10 and
    DM11 (after the estimates, which carry them too; only with the CROWN shifts) and the ttbar contamination of the
    embedded sample (+-10 % of genuine ttbar)."""
    genuine_and_lepton_fakes = backgrounds_in(processes, GENUINE) + backgrounds_in(processes, LEPTON_FAKE)
    every_mc = genuine_and_lepton_fakes + backgrounds_in(processes, JET_FAKE)
    output = jet_fakes_output(switches.jet_fakes)
    if switches.jet_fakes == "ff":
        result = (DataMinus(output, "anti_iso", genuine_and_lepton_fakes),)
    else:
        result = (ABCD(output, "abcd_anti_iso", "abcd_same_sign", "abcd_same_sign_anti_iso", every_mc),)
    if switches.diagnostics:
        result += diagnostic_estimators(name, every_mc)
    if switches.embedded(name) and switches.shape_systematics and switches.crown_shifts:
        result += embedding_variation_sums(name, era)
    if switches.embedded(name):
        result += (TemplateShift(embedding_ttbar_nuisance(era), EMBEDDED, SPLITS["TT"]["T"], 0.1),)
    return result


def column_variations(name: str, era: str, switches: TauSwitches) -> tuple:
    """The embedding tau shifts (CROWN shifts, not in a nominal-only production) and the fake-factor friend shifts."""
    if not switches.shape_systematics:
        return ()
    embedding = embedding_variations(name, era) if switches.embedded(name) and switches.crown_shifts else ()
    return embedding + (ff_variations(name, era, switches.wjets) if switches.jet_fakes == "ff" else ())


def fit_categories(switches: TauSwitches) -> tuple[Category, ...]:
    if switches.fit_categories == "control":
        return nominal_category()
    return categories() if switches.nn_friend else ()


def channel(name: str, era: str, switches: TauSwitches, channel_samples: tuple[Sample, ...]) -> Channel:
    table = available(tau_processes(name, era, switches.jet_fakes, switches.embedded(name), switches.dy_correction), channel_samples)
    groups = {p.group for p in table}
    return Channel(
        name=name,
        samples=tuple(s for s in channel_samples if s.group in groups),
        skim=cuts.skim_cuts(name, era, switches.control_regions),
        cuts=cuts.baseline_cuts(name, era),
        processes=table,
        regions=cuts.regions(name, switches.jet_fakes, switches.ff_type) + (cuts.diagnostic_regions(name) if switches.control_regions else ()),
        categories=fit_categories(switches),
        variables=control_variables(switches.nn_friend, switches.diagnostics),
        variations=btag_variations(era) + prefiring_variations() + column_variations(name, era, switches),
        estimators=estimators(name, era, table, switches),
        keep_columns=("event", "run", "lumi"),
    )


def borrowed_processes(channels: Iterable[Channel]) -> tuple[str, ...]:
    """The processes whose samples are all borrowed from another era (Sample.era)."""
    result = {}
    for channel in channels:
        for process in channel.processes:
            own = channel.samples_of(process.group)
            if own and all(s.era is not None for s in own) and process.role in ("signal", "background"):
                result[process.name] = None
    return tuple(result)


def build(config: RunConfig) -> Analysis:
    era = check_era(config.era)
    switches = parse(TauSwitches, config.switches)
    if not config.channels or set(config.channels) - set(TAU_CHANNELS + CONTROL_CHANNELS):
        raise AnalysisError(f"the tau analysis supports the channels {', '.join(TAU_CHANNELS)} and the control channels {', '.join(CONTROL_CHANNELS)}")
    switches.check_embedding(config.channels)
    by_channel = samples(sample_database(config), switches.sample_lists, config.channels, era)
    channels = {
        name: dilepton.top_control_channel(by_channel[name], era, switches.dy_correction) if name in CONTROL_CHANNELS else channel(name, era, switches, by_channel[name])
        for name in config.channels
    }
    tau_channels = [c for name, c in channels.items() if name not in CONTROL_CHANNELS]
    return Analysis(
        name=f"hh_bbtautau_{era}_v15",
        era=era,
        lumi_pb=LUMI_PB[era],
        signal=SIGNAL,
        channels=channels,
        control_channels=tuple(name for name in config.channels if name in CONTROL_CHANNELS),
        # the channels can differ in their processes
        lnn=lnn(process_union(channels.values()), jet_fakes_output(switches.jet_fakes), era, borrowed_processes(channels.values())),
        lnn_variations=(PREFIRING_NUISANCE,),
        style=style(era),
        ml=ml_config(process_union(tau_channels), switches.jet_fakes),  # no NN training on a control channel
    )
