"""Assemble the HH->bbtautau Analysis of the tau channels (and the ttbar control channel em) of one era from the run
configuration switches, with its ML export (ml.py)."""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path

from shapesmith.config import RunConfig
from shapesmith.model import ABCD, Analysis, AnalysisError, Category, Channel, CrossSection, DataMinus, Process, Sample, TemplateShift

from bbtautau_shapesmith import cuts, dilepton
from bbtautau_shapesmith.constants import (
    BTAG_BINS, CONTROL_CHANNELS, LT_CHANNELS, LUMI_PB, NN_CLASS_COLUMN, NN_SCORE_COLUMN, PREFIRING_NUISANCE, TAU_CHANNELS, check_era,
)
from bbtautau_shapesmith.ml import ML_CLASSES, ml_config, raw_count_groups
from bbtautau_shapesmith.nn_validation import FASTMTT_INPUTS, pseudo_shift_names, pseudo_shift_variations, validation_variables
from bbtautau_shapesmith.processes import (
    EMBEDDED, GENUINE, JET_FAKE, LEPTON_FAKE, NODE_SM, SIGNAL, SIGNAL_XS_FB, SIGNAL_XS_NUISANCES, SPLITS, available, backgrounds_in, tau_processes,
)
from bbtautau_shapesmith.samples import sample_database, samples
from bbtautau_shapesmith.style import style
from bbtautau_shapesmith.switches import TauSwitches, parse
from bbtautau_shapesmith.systematics import (
    HEAVY_FLAVOUR_WEIGHT, SIGNAL_THEORY_WEIGHT, Study, btag_variations, embedding_id_variations, embedding_ttbar_nuisance, embedding_variation_sums,
    embedding_variations, ff_dm_nonclosure_variations, ff_variant, ff_variations, heavy_flavour_variations, lnn, prefiring_variations,
    signal_theory_variations, ttbar_shape_variations, with_weights,
)
from bbtautau_shapesmith.variables import categories, control_variables, nn_score_blinding, nominal_category

def process_union(channels: Iterable[Channel]) -> tuple[Process, ...]:
    """The processes of all channels, each name once in channel order; a process that is auxiliary in one channel and
    a background in another (TTT with and without embedding) takes its background role."""
    union = {}
    for channel in channels:
        for process in channel.processes:
            if process.name not in union or union[process.name].role == "auxiliary":
                union[process.name] = process
    return tuple(union.values())


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
    """The embedding tau shifts (CROWN shifts, not in a nominal-only production) and the fake-factor friend shifts; with
    study_uncertainties the embedding tau ID as weight factors in a nominal-only production and the decay-mode
    non-closure of the fake factors."""
    if not switches.shape_systematics:
        return ()
    embedding = embedding_variations(name, era) if switches.embedded(name) and switches.crown_shifts else ()
    if switches.study_uncertainties and switches.embedded(name) and not switches.crown_shifts and name in LT_CHANNELS:
        embedding += embedding_id_variations(name, era)
    fakes = ff_variations(name, era, switches.wjets) if switches.jet_fakes == "ff" or switches.ff_transfer else ()
    if switches.study_uncertainties and switches.jet_fakes == "ff":
        fakes += ff_dm_nonclosure_variations(name, era, ff_variant(switches.wjets))
    pseudo = pseudo_shift_variations(name, era, switches.embedded(name)) if switches.pseudo_shifts else ()
    return embedding + fakes + pseudo


def study(switches: TauSwitches, channels: Iterable[str]) -> Study | None:
    """The switch study_uncertainties for lnn(); None without it."""
    if not switches.study_uncertainties:
        return None
    embedded = tuple(name for name in channels if name in TAU_CHANNELS and switches.embedded(name))
    return Study(embedded, ff_variant(switches.wjets) if switches.jet_fakes == "ff" else None, switches.crown_shifts)


def nn_classes(switches: TauSwitches) -> tuple[str, ...]:
    """The NN classes in the order of the friend's predicted_class: nn_classes, else those of the label profile of the
    export, by default the profile of the jet-fake mode (thesis6 with mc, v7b5 with ff)."""
    if switches.nn_classes is not None:
        return tuple(switches.nn_classes)
    return ML_CLASSES[switches.ml_labels or {"mc": "thesis6", "ff": "v7b5"}[switches.jet_fakes]]


def check_manifest(manifest: str, channels: Iterable[str], classes: tuple[str, ...]) -> None:
    """The model manifest (nn_manifest, {channel} filled per tau channel) lists exactly the NN classes of the run."""
    for name in channels:
        path = Path(manifest.replace("{channel}", name))
        if not path.is_file():
            raise AnalysisError(f"nn_manifest: {path} does not exist")
        listed = json.loads(path.read_text()).get("classes")
        if listed is None or tuple(listed) != classes:
            raise AnalysisError(f"nn_manifest: {path} lists the classes {listed}, the run has {list(classes)} (nn_classes or the label profile, in the order of predicted_class)")


def nn_friend_columns(switches: TauSwitches) -> tuple[str, ...]:
    """The NN friend columns a tau skim keeps besides those the run reads: the class and score of every model of the
    friend production (nn_friend_prefixes), so that the runs of all its models share one skim."""
    if not switches.nn_friend:
        return ()
    return tuple(f"{prefix}{column}" for prefix in switches.nn_friend_prefixes for column in (NN_CLASS_COLUMN, NN_SCORE_COLUMN))


def fit_categories(switches: TauSwitches, processes: tuple[Process, ...]) -> tuple[Category, ...]:
    if switches.fit_categories == "control":
        return nominal_category()
    if switches.fit_categories != "nn":
        return ()
    groups = raw_count_groups(processes, switches.jet_fakes)
    return categories(nn_classes(switches), switches.nn_categories, switches.nn_binning, switches.nn_prefix, groups, clip=(jet_fakes_output(switches.jet_fakes),))


def channel(name: str, era: str, switches: TauSwitches, channel_samples: tuple[Sample, ...]) -> Channel:
    """A tau channel; with the NN friend the control variable NN_score leaves out the events of the signal category
    (variables.nn_score_blinding)."""
    table = available(tau_processes(name, era, switches.jet_fakes, switches.embedded(name), switches.dy_correction), channel_samples)
    groups = {p.group for p in table}
    baseline = cuts.baseline_cuts(name, era)
    regions = (
        cuts.regions(name, switches.jet_fakes, switches.ff_type)
        + (cuts.diagnostic_regions(name) if switches.control_regions else ())
        + ((cuts.fake_factor_region(name, switches.ff_type),) if switches.ff_transfer else ())
    )
    estimates = estimators(name, era, table, switches)
    blinded = nn_score_blinding(nn_classes(switches), switches.nn_prefix, baseline, regions, estimates) if switches.nn_friend else None
    variables = (
        validation_variables(name, switches.fastmtt_friend)
        if switches.nn_validation
        else control_variables(switches.nn_friend, switches.diagnostics, switches.fastmtt_friend, switches.nn_prefix, blinded)
    )
    return Channel(
        name=name,
        samples=tuple(s for s in channel_samples if s.group in groups),
        skim=cuts.skim_cuts(name, era, switches.control_regions),
        cuts=baseline,
        processes=table,
        regions=regions,
        categories=fit_categories(switches, table),
        variables=variables,
        variations=btag_variations(era) + prefiring_variations() + column_variations(name, era, switches),
        estimators=estimates,
        keep_columns=("event", "run", "lumi", *nn_friend_columns(switches)) + (FASTMTT_INPUTS if switches.nn_validation else ()),
    )


def with_stage_variations(channel: Channel, era: str, switches: TauSwitches) -> Channel:
    """A tau or control channel with the weight variations of the NN systematics stages (nn_limit_spec.md 8):
    ttbar_shape on the processes with the top-pT weight, heavy_flavour on the simulated backgrounds and signal_theory on
    the signal, each of the last two with the nominal factor 1.0 it replaces (systematics.with_weights)."""
    processes, variations = channel.processes, ()
    if switches.ttbar_shape:
        variations += ttbar_shape_variations(era)
    if switches.heavy_flavour:
        processes = with_weights(processes, {p.name for p in processes if p.role == "background" and p.group != EMBEDDED}, HEAVY_FLAVOUR_WEIGHT)
        variations += heavy_flavour_variations(era)
    if switches.signal_theory:
        processes = with_weights(processes, {p.name for p in processes if p.role == "signal"}, SIGNAL_THEORY_WEIGHT)
        variations += signal_theory_variations(era, switches.signal_theory)
    return dataclasses.replace(channel, processes=processes, variations=channel.variations + variations) if variations else channel


def symmetrised_ff_shifts(channels: Iterable[Channel], era: str, switches: TauSwitches) -> tuple[str, ...]:
    """With symmetrise_ff_shifts the names (without the direction) of the fake-factor friend shifts of the channels,
    which the datacards symmetrise where one-sided or strongly asymmetric; the other variations stay as they are."""
    if not (switches.symmetrise_ff_shifts and switches.shape_systematics):
        return ()
    names = {v.name[: -len("Up")] for c in channels for v in ff_variations(c.name, era, switches.wjets) if v.name.endswith("Up")}
    return tuple(sorted(names))


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
    if switches.nn_manifest is not None:
        check_manifest(switches.nn_manifest, [name for name in config.channels if name in TAU_CHANNELS], nn_classes(switches))
    signal = NODE_SM if switches.signal_sample == "node_sm" else SIGNAL
    by_channel = samples(sample_database(config), switches.sample_lists, config.channels, era, signal)
    channels = {
        name: with_stage_variations(
            dilepton.top_control_channel(by_channel[name], era, switches.dy_correction, switches.fit_categories == "nn") if name in CONTROL_CHANNELS else channel(name, era, switches, by_channel[name]), era, switches,
        )
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
        lnn=lnn(process_union(channels.values()), jet_fakes_output(switches.jet_fakes), era, borrowed_processes(channels.values()), study(switches, config.channels), switches.dy_correction),
        lnn_variations=(PREFIRING_NUISANCE,),
        symmetrised_variations=symmetrised_ff_shifts(tau_channels, era, switches) + pseudo_shift_names(era, any(switches.embedded(c.name) for c in tau_channels), switches.pseudo_shifts),
        style=style(era),
        # no NN training on a control channel
        ml=ml_config(process_union(tau_channels), switches),
        signal_cross_section=CrossSection("ggF HH", SIGNAL_XS_FB, SIGNAL_XS_NUISANCES),  # ggF only (N23)
    )
