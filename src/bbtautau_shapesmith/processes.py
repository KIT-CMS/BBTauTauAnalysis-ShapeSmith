"""The process tables per channel.

Tau channels: which processes exist depends on the switches jet_fakes (ff|mc) and embedding. Gen-match splits: T =
genuine di-tau, L = lepton fakes, J = jet -> tau_h fakes. In fake-factor mode the J parts and W are replaced by the
estimated jetFakes process; with embedding the T parts are replaced by EMB, and the genuine ttbar part stays as the
auxiliary template of the embedding ttbar-contamination variation.
Light-dilepton channels: every MC group unsplit and no jet-fake estimate (W is plain MC).
Every DY process carries the DY correction with dy_correction: njets_btag.
Optional groups (VH -> tautau, the node_SM cross-check of the signal) have a process only where the sample lists have
samples of them (v7 spec 6.3); `available` drops the others.
"""
from __future__ import annotations

from collections.abc import Iterable

from shapesmith.model import Process, Sample, Selection

from bbtautau_shapesmith.cuts import genmatch_cuts
from bbtautau_shapesmith.weights import dilepton_mc_weights, embedding_weights, group_weights, mc_weights

SIGNAL = "HH2B2Tau"  # the powheg NLO ggF HH -> bbtautau, kappa_lambda = 1 (E23)
# sigma(ggF HH) at kappa_lambda = 1 in fb, the unit of the limits on the cross section (spec N23): r x SIGNAL_XS_FB; the
# sample cross section 0.002221 pb is this times 2 BR(H->bb) BR(H->tautau). Its theory nuisances (scale + THnum + m_t,
# PDF + alpha_s; systematics.lnn) are frozen in that limit, the BR and the acceptance and shape uncertainties stay.
SIGNAL_XS_FB = 30.40
SIGNAL_XS_NUISANCES = ("QCDscale_HH", "PDF_alphas_HH")
NODE_SM = "HH2B2Tau_nodeSM"  # the madgraph node_SM sample: the cross-check of the signal (an auxiliary process)
VH_TAUTAU = "VH_htt"  # ZH and W+-H with H -> tautau (E23)
OPTIONAL_GROUPS = frozenset({VH_TAUTAU, NODE_SM})
EMBEDDED = "EMB"
SPLITS = {
    "DY": {"T": "ZTT", "L": "ZL", "J": "ZJ"},
    "TT": {"T": "TTT", "L": "TTL", "J": "TTJ"},
    "ST": {"T": "STT", "L": "STL", "J": "STJ"},
    "VV": {"T": "VVT", "L": "VVL", "J": "VVJ"},
    "TTV": {"T": "TTVT", "L": "TTVL", "J": "TTVJ"},
    "EWK": {"T": "EWKT", "L": "EWKL", "J": "EWKJ"},
}
SINGLE_HIGGS = {"ggH": "ggH125", "qqH": "qqH125", "ttH": "ttH125", "VH": "VH125", VH_TAUTAU: VH_TAUTAU}  # group -> process
PLOT_GROUP = {"DY": "Z", "TT": "TT", "ST": "ST", "VV": "VV", "TTV": "rare", "EWK": "rare", "W": "rare", **dict.fromkeys(SINGLE_HIGGS, "rare")}

# hadron flavour of the GenJet of the leading-score b-pair jet (MC only; 246 = no GenJet, light as in the BTV convention)
HADRON_FLAVOURS = {"b": "(genjet_hadFlavour_1 == 5)", "c": "(genjet_hadFlavour_1 == 4)", "l": "(genjet_hadFlavour_1 != 5) & (genjet_hadFlavour_1 != 4)"}

# analysis vocabulary for the estimates and the ML labels
GENUINE = frozenset({EMBEDDED, *(parts["T"] for parts in SPLITS.values())})
LEPTON_FAKE = frozenset(parts["L"] for parts in SPLITS.values())
JET_FAKE = frozenset({"W", *(parts["J"] for parts in SPLITS.values())})


def _mc(name: str, group: str, weights: dict[str, str], era: str, cuts: dict[str, str] | None = None, role: str = "background", dy_correction: str = "none") -> Process:
    return Process(name, group, role, PLOT_GROUP[group], Selection(cuts=cuts or {}, weights={**weights, **group_weights(group, era, dy_correction)}))


def _data_and_signal(weights: dict[str, str]) -> list[Process]:
    return [Process("data", "data", "data", "data"), Process(SIGNAL, SIGNAL, "signal", "signal", Selection(weights=weights))]


def tau_processes(channel: str, era: str, jet_fakes: str, embedding: bool, dy_correction: str = "none") -> tuple[Process, ...]:
    mc, genmatch = mc_weights(channel, era), genmatch_cuts(channel)
    result = _data_and_signal(mc) + [Process(NODE_SM, NODE_SM, "auxiliary", "signal", Selection(weights=mc))]
    if embedding:
        result.append(Process(EMBEDDED, EMBEDDED, "background", EMBEDDED, Selection(cuts={"genmatch": genmatch["T"]}, weights=embedding_weights(channel, era))))
    parts = ("L",) + (() if embedding else ("T",)) + (("J",) if jet_fakes == "mc" else ())
    for group, names in SPLITS.items():
        result += [_mc(name, group, mc, era, {"genmatch": genmatch[part]}, dy_correction=dy_correction) for part, name in names.items() if part in parts]
    if embedding:
        result.append(_mc(SPLITS["TT"]["T"], "TT", mc, era, {"genmatch": genmatch["T"]}, role="auxiliary"))
    if jet_fakes == "mc":
        result.append(_mc("W", "W", mc, era))
    result += [_mc(name, group, mc, era) for group, name in SINGLE_HIGGS.items()]
    return tuple(result)


def dilepton_processes(channel: str, era: str, dy_correction: str = "none", flavours: bool = False) -> tuple[Process, ...]:
    """With flavours, the auxiliary flavour templates of DY (mm, ee) or ttbar (em), e.g. DY_b, DY_c and DY_l."""
    mc = dilepton_mc_weights(channel, era)
    result = _data_and_signal(mc)
    result += [_mc(group, group, mc, era, dy_correction=dy_correction) for group in (*SPLITS, "W")]
    result += [_mc(name, group, mc, era) for group, name in SINGLE_HIGGS.items()]
    if flavours:
        group = "TT" if channel == "em" else "DY"
        result += [_mc(f"{group}_{flavour}", group, mc, era, {"flavour": cut}, role="auxiliary", dy_correction=dy_correction) for flavour, cut in HADRON_FLAVOURS.items()]
    return tuple(result)


def available(processes: Iterable[Process], samples: Iterable[Sample]) -> tuple[Process, ...]:
    """The processes without those of an optional group that has no samples."""
    groups = {s.group for s in samples}
    return tuple(p for p in processes if p.group not in OPTIONAL_GROUPS or p.group in groups)


def backgrounds_in(processes: tuple[Process, ...], names: frozenset[str]) -> tuple[str, ...]:
    """The background processes of the table with one of the given names, in table order."""
    return tuple(p.name for p in processes if p.role == "background" and p.name in names)
