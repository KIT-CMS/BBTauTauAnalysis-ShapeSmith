"""Light-dilepton controls: Z+jets in ee/mm and top in em with unsplit MC and no tau-fake estimate.

Leg order and triggers follow CROWN's 2018 SM profile (em: electron leg 1, muon leg 2; mm/ee: leading pT first).
A separate Analysis from the tau channels; the two share samples, style and the b-tag variations.
"""
from __future__ import annotations

from shapesmith.config import RunConfig
from shapesmith.model import Analysis, AnalysisError, Channel, Region, Sample

from bbtautau_shapesmith.constants import BTAG_BINS, DILEPTON_CHANNELS, ERA, LUMI_PB
from bbtautau_shapesmith.processes import SIGNAL, dilepton_processes
from bbtautau_shapesmith.samples import sample_database, samples
from bbtautau_shapesmith.style import style
from bbtautau_shapesmith.switches import DileptonSwitches, parse
from bbtautau_shapesmith.systematics import btag_variations
from bbtautau_shapesmith.variables import control_variables

PLOT_VARIABLES = (
    "yield", "pt_1", "pt_2", "eta_1", "eta_2", "phi_1", "phi_2", "iso_1", "iso_2",
    "m_vis", "pt_vis", "met", "metphi", "n_jets", "n_bjets", "jpt_1", "jpt_2",
    "jeta_1", "jeta_2", "mjj", "bpair_pt_1", "bpair_pt_2", "bpair_m_inv",
    "bpair_deltaR", "bpair_btag_value_1", "bpair_btag_value_2",
)


def baseline_cuts(channel: str) -> dict[str, str]:
    return {
        "os": "(q_1 * q_2) < 0",
        "lepton_iso": "(iso_1 < 0.15) & (iso_2 < 0.15)",
        "extraelec_veto": "extraelec_veto < 0.5", "extramuon_veto": "extramuon_veto < 0.5",
        # Match the effective offline thresholds inside CROWN's trigger flags.
        "pt_selection": {"em": "(pt_1 > 15) & (pt_2 > 26)",
                         "mm": "(pt_1 > 26) & (pt_2 > 15)",
                         "ee": "(pt_1 > 34) & (pt_2 > 15)"}[channel],
        "trigger": "trg_single_ele32 > 0.5" if channel == "ee" else "trg_single_mu24 > 0.5",
        "mass_window": "True" if channel == "em" else "(m_vis > 70) & (m_vis < 110)",
        "jets": "n_jets >= 2" if channel == "em" else "True",
        "b_tagging": "True",
    }


def regions() -> tuple[Region, ...]:
    btag = tuple(Region(name, replace_cuts={"jets": "n_jets >= 2", "b_tagging": cut}) for name, cut in BTAG_BINS.items())
    return btag + (Region("same_sign", replace_cuts={"os": "(q_1 * q_2) > 0"}),)


def channel(name: str, channel_samples: tuple[Sample, ...]) -> Channel:
    table = dilepton_processes(name)
    groups = {p.group for p in table}
    cuts = baseline_cuts(name)
    variables = control_variables()
    return Channel(
        name=name,
        samples=tuple(s for s in channel_samples if s.group in groups),
        skim={cut: expr for cut, expr in cuts.items() if cut != "os"},  # both charges, for the same-sign validation
        cuts=cuts,
        processes=table,
        regions=regions(),
        variables={variable: variables[variable] for variable in PLOT_VARIABLES},
        variations=btag_variations(("correlated", "uncorrelated")),  # the summary scheme, not its decomposition
        keep_columns=("event", "run", "lumi"),
    )


def build(config: RunConfig) -> Analysis:
    switches = parse(DileptonSwitches, config.switches)
    if config.era != ERA or not config.channels or set(config.channels) - set(DILEPTON_CHANNELS):
        raise AnalysisError(f"the dilepton analysis supports era {ERA} and the channels {', '.join(DILEPTON_CHANNELS)}")
    by_channel = samples(sample_database(config), switches.sample_lists, config.channels)
    return Analysis(
        name="hh_bbtautau_2018_dilepton_controls",
        era=ERA,
        lumi_pb=LUMI_PB,
        signal=SIGNAL,
        channels={name: channel(name, by_channel[name]) for name in config.channels},
        style=style(),
    )
