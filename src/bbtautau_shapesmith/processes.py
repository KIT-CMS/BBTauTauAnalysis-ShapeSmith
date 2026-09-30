"""The process tables per channel.

Tau channels: which processes exist depends on the switches jet_fakes (ff|mc) and embedding. Gen-match splits: T =
genuine di-tau, L = lepton fakes, J = jet -> tau_h fakes. In fake-factor mode the J parts and W are replaced by the
estimated jetFakes process; with embedding the T parts are replaced by EMB.
Light-dilepton channels: every MC group unsplit and no jet-fake estimate (W is plain MC).
"""
from __future__ import annotations

from shapesmith.model import Process, Selection

from bbtautau_shapesmith.cuts import genmatch_cuts
from bbtautau_shapesmith.weights import TOP_PT, dilepton_mc_weights, embedding_weights, mc_weights

SIGNAL = "HH2B2Tau"
EMBEDDED = "EMB"
SPLITS = {
    "DY": {"T": "ZTT", "L": "ZL", "J": "ZJ"},
    "TT": {"T": "TTT", "L": "TTL", "J": "TTJ"},
    "ST": {"T": "STT", "L": "STL", "J": "STJ"},
    "VV": {"T": "VVT", "L": "VVL", "J": "VVJ"},
    "TTV": {"T": "TTVT", "L": "TTVL", "J": "TTVJ"},
    "EWK": {"T": "EWKT", "L": "EWKL", "J": "EWKJ"},
}
SINGLE_HIGGS = {"ggH": "ggH125", "qqH": "qqH125", "ttH": "ttH125", "VH": "VH125"}  # group -> process
PLOT_GROUP = {"DY": "Z", "TT": "TT", "ST": "ST", "VV": "VV", "TTV": "rare", "EWK": "rare", "W": "rare", **dict.fromkeys(SINGLE_HIGGS, "rare")}

# analysis vocabulary for the estimates and the ML labels
GENUINE = frozenset({EMBEDDED, *(parts["T"] for parts in SPLITS.values())})
LEPTON_FAKE = frozenset(parts["L"] for parts in SPLITS.values())
JET_FAKE = frozenset({"W", *(parts["J"] for parts in SPLITS.values())})


def _mc(name: str, group: str, weights: dict[str, str], cuts: dict[str, str] | None = None, role: str = "background") -> Process:
    process_weights = {**weights, **TOP_PT} if group == "TT" else weights
    return Process(name, group, role, PLOT_GROUP[group], Selection(cuts=cuts or {}, weights=process_weights))


def _data_and_signal(weights: dict[str, str]) -> list[Process]:
    return [Process("data", "data", "data", "data"), Process(SIGNAL, SIGNAL, "signal", "signal", Selection(weights=weights))]


def tau_processes(channel: str, jet_fakes: str, embedding: bool) -> tuple[Process, ...]:
    mc, genmatch = mc_weights(channel), genmatch_cuts(channel)
    result = _data_and_signal(mc)
    if embedding:
        result.append(Process(EMBEDDED, EMBEDDED, "background", EMBEDDED, Selection(cuts={"genmatch": genmatch["T"]}, weights=embedding_weights(channel))))
    parts = ("L",) + (() if embedding else ("T",)) + (("J",) if jet_fakes == "mc" else ())
    for group, names in SPLITS.items():
        result += [_mc(name, group, mc, {"genmatch": genmatch[part]}) for part, name in names.items() if part in parts]
    if jet_fakes == "mc":
        result.append(_mc("W", "W", mc))
    result += [_mc(name, group, mc) for group, name in SINGLE_HIGGS.items()]
    return tuple(result)


def dilepton_processes(channel: str) -> tuple[Process, ...]:
    mc = dilepton_mc_weights(channel)
    result = _data_and_signal(mc)
    result += [_mc(group, group, mc) for group in (*SPLITS, "W")]
    result += [_mc(name, group, mc) for group, name in SINGLE_HIGGS.items()]
    return tuple(result)


def backgrounds_in(processes: tuple[Process, ...], names: frozenset[str]) -> tuple[str, ...]:
    """The background processes of the table with one of the given names, in table order."""
    return tuple(p.name for p in processes if p.role == "background" and p.name in names)
