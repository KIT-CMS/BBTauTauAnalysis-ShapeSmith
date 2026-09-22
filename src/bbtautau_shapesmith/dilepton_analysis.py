"""Light-dilepton controls: Z+jets in ee/mm and top in em with unsplit MC and no tau-fake estimate.

A separate Analysis from the tau channels because the ShapeSmith core keeps the process table, the estimator
and the control variables per Analysis; the two share samples, style and the b-tag variations.
"""
from shapesmith.config import RunConfig
from shapesmith.model import Analysis, AnalysisError

from bbtautau_shapesmith.constants import DILEPTON_CHANNELS, ERA, LUMI_PB
from bbtautau_shapesmith.dilepton_selection import channel_definition
from bbtautau_shapesmith.processes import SIGNAL, control_variables, dilepton_processes
from bbtautau_shapesmith.samples import DEFAULT_SAMPLE_LIST, sample_database, samples
from bbtautau_shapesmith.systematics import btag_variations, style

PLOT_VARIABLES = (
    "yield", "pt_1", "pt_2", "eta_1", "eta_2", "phi_1", "phi_2", "iso_1", "iso_2",
    "m_vis", "pt_vis", "met", "metphi", "n_jets", "n_bjets", "jpt_1", "jpt_2",
    "jeta_1", "jeta_2", "mjj", "bpair_pt_1", "bpair_pt_2", "bpair_m_inv",
    "bpair_deltaR", "bpair_btag_value_1", "bpair_btag_value_2",
)


def build(config: RunConfig) -> Analysis:
    unknown = set(config.switches) - {"sample_list"}
    if unknown:
        raise AnalysisError(f"unsupported light-dilepton switches: {sorted(unknown)}; only sample_list is supported")
    if config.era != ERA or not config.channels or set(config.channels) - set(DILEPTON_CHANNELS):
        raise AnalysisError(f"dilepton_analysis supports era {ERA} and the channels {', '.join(DILEPTON_CHANNELS)}")
    variables = control_variables()
    return Analysis(
        name="hh_bbtautau_2018_dilepton_controls",
        era=ERA,
        lumi_pb=LUMI_PB,
        channels={channel: channel_definition(channel) for channel in config.channels},
        samples=samples(sample_database(config), config.switches.get("sample_list", DEFAULT_SAMPLE_LIST)),
        processes=dilepton_processes(),
        signal=SIGNAL,
        control_variables={name: variables[name] for name in PLOT_VARIABLES},
        weight_variations=btag_variations(("correlated", "uncorrelated")),  # the summary scheme, not its decomposition
        style=style(),
    )
