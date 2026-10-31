"""Light-dilepton controls: Z+jets in ee/mm and top in em with unsplit MC and no tau-fake estimate.

Leg order follows CROWN's SM profile (em: electron leg 1, muon leg 2; mm/ee: leading pT first), the triggers and their
offline thresholds triggers.TRIGGER of the era (v7 spec 6.2: em and mm the muon trigger, ee the electron trigger; 2016
the triggering lepton within |eta| < 2.1), 2018 vetoes the HEM 15/16 events (E29). A separate Analysis from the tau
channels; the two share samples, style and the b-tag and prefiring variations. em is also the ttbar control channel of
the tau analysis (top_control_channel).
"""
from __future__ import annotations

import dataclasses

from shapesmith.config import RunConfig
from shapesmith.model import Analysis, AnalysisError, Category, Channel, Region, Sample

from bbtautau_shapesmith.constants import BTAG_BINS, DILEPTON_CHANNELS, EVENT_KEY, LUMI_PB, NJETS_BINS, PREFIRING_NUISANCE, check_era
from bbtautau_shapesmith.cuts import B_TAGGING, era_cuts
from bbtautau_shapesmith.nn_validation import FASTMTT_INPUTS, dilepton_validation_variables
from bbtautau_shapesmith.processes import SIGNAL, available, dilepton_processes
from bbtautau_shapesmith.samples import sample_database, samples
from bbtautau_shapesmith.style import style
from bbtautau_shapesmith.switches import DileptonSwitches, parse
from bbtautau_shapesmith.systematics import btag_variations, prefiring_variations
from bbtautau_shapesmith.triggers import TRIGGER
from bbtautau_shapesmith.variables import DIAGNOSTIC_VARIABLES, control_variables

PLOT_VARIABLES = (
    "yield", "pt_1", "pt_2", "eta_1", "eta_2", "phi_1", "phi_2", "iso_1", "iso_2",
    "m_vis", "pt_vis", "met", "metphi", "n_jets", "n_bjets", "jpt_1", "jpt_2",
    "jeta_1", "jeta_2", "mjj", "bpair_pt_1", "bpair_pt_2", "bpair_m_inv",
    "bpair_deltaR", "bpair_btag_value_1", "bpair_btag_value_2",
)


def baseline_cuts(channel: str, era: str, min_jets: int = 0) -> dict[str, str]:
    """The selection of a channel; `min_jets` (the input validation: 2) requires the jets in mm and ee as well."""
    trigger = TRIGGER[era][channel]
    return {
        "os": "(q_1 * q_2) < 0",
        "lepton_iso": "(iso_1 < 0.15) & (iso_2 < 0.15)",
        "extraelec_veto": "extraelec_veto < 0.5", "extramuon_veto": "extramuon_veto < 0.5",
        # the effective offline thresholds inside CROWN's trigger flags
        "pt_selection": trigger.thresholds,
        "trigger": trigger.cut,
        "mass_window": "True" if channel == "em" else "(m_vis > 70) & (m_vis < 110)",
        "jets": "n_jets >= 2" if channel == "em" else (f"n_jets >= {min_jets}" if min_jets else "True"),
        "b_tagging": "True",
        **era_cuts(era),
    }


def regions(diagnostics: bool = False) -> tuple[Region, ...]:
    """The b-tag bins (>= 2 jets) and same sign; with diagnostics the jet multiplicities with 0 and >= 1 b tags."""
    btag = tuple(Region(name, replace_cuts={"jets": "n_jets >= 2", "b_tagging": cut}) for name, cut in BTAG_BINS.items())
    result = btag + (Region("same_sign", replace_cuts={"os": "(q_1 * q_2) > 0"}),)
    if diagnostics:
        tags = {"btag0": "n_bjets == 0", "btag1p": "n_bjets >= 1"}
        result += tuple(Region(f"{jets}_{tag}", replace_cuts={"jets": jet_cut, "b_tagging": tag_cut}) for jets, jet_cut in NJETS_BINS.items() for tag, tag_cut in tags.items())
    return result


def channel(name: str, era: str, channel_samples: tuple[Sample, ...], dy_correction: str = "none", diagnostics: bool = False, nn_validation: bool = False) -> Channel:
    """A dilepton channel; with nn_validation >= 2 jets and the variables of the input validation (nn_validation.py)."""
    table = available(dilepton_processes(name, era, dy_correction, flavours=diagnostics), channel_samples)
    groups = {p.group for p in table}
    cuts = baseline_cuts(name, era, 2 if nn_validation else 0)
    variables = control_variables(diagnostics=diagnostics)
    shown = {variable: variables[variable] for variable in PLOT_VARIABLES + (DIAGNOSTIC_VARIABLES if diagnostics else ())}
    return Channel(
        name=name,
        samples=tuple(s for s in channel_samples if s.group in groups),
        skim={cut: expr for cut, expr in cuts.items() if cut != "os"},  # both charges, for the same-sign validation
        cuts=cuts,
        processes=table,
        regions=regions(diagnostics),
        variables=dilepton_validation_variables() if nn_validation else shown,
        variations=btag_variations(era) + prefiring_variations(),  # the variations of the tau channels, so that they correlate
        keep_columns=("event", "run", "lumi") + (FASTMTT_INPUTS if nn_validation else ()),
    )


def top_control_channel(channel_samples: tuple[Sample, ...], era: str, dy_correction: str = "none", btag_split: bool = False) -> Channel:
    """em as the ttbar control channel of the tau analysis: the em selection (>= 2 jets) with the b tag of the tau
    baseline as its one category, counted in one bin, without further regions; with btag_split (the NN categories,
    SH4, N12) the categories em_tt_btag1 and em_tt_btag2 (n_bjets == 1, == 2) instead, without n_bjets >= 3."""
    yields = control_variables()["yield"]
    categories = (Category("em_tt", B_TAGGING, yields),)
    if btag_split:
        categories = tuple(Category(f"em_tt_{name}", f"{B_TAGGING} & ({BTAG_BINS[name]})", yields) for name in ("btag1", "btag2"))
    return dataclasses.replace(channel("em", era, channel_samples, dy_correction), regions=(), categories=categories)


def build(config: RunConfig) -> Analysis:
    era = check_era(config.era)
    switches = parse(DileptonSwitches, config.switches)
    if not config.channels or set(config.channels) - set(DILEPTON_CHANNELS):
        raise AnalysisError(f"the dilepton analysis supports the channels {', '.join(DILEPTON_CHANNELS)}")
    by_channel = samples(sample_database(config), switches.sample_lists, config.channels, era)
    return Analysis(
        name=f"hh_bbtautau_{era}_dilepton_controls",
        era=era,
        lumi_pb=LUMI_PB[era],
        signal=SIGNAL,
        channels={name: channel(name, era, by_channel[name], switches.dy_correction, switches.diagnostics, switches.nn_validation) for name in config.channels},
        lnn_variations=(PREFIRING_NUISANCE,),
        style=style(era),
        event_key=EVENT_KEY,
    )
