"""The fit categories (NN classes or one nominal category) and the control variables with their binning.

NN categories (NN spec 6.2, 6.3; switches nn_categories, nn_binning): one per class of the model, the events whose
predicted_class is the class index (the class order of the model and of the ML export), the discriminant its score
predicted_max_value (in the HH category P(HH)), filled in 1000 bins on [0, 1] whose edges datacards resolve from the
nominal templates (shapesmith ScoreBinning):
- thesis: the categories of the master's thesis, argmax only (R3-R5);
- k1: argmax with n_bjets >= 3 cut out (N12, the baseline has n_bjets >= 1);
- k2: k1 with the HH category split into HH2B2Tau_btag1 (n_bjets == 1) and HH2B2Tau_btag2 (n_bjets == 2), N10.
Binning: data_blind, the rules of shapesmith.nn_binning (HH: 50 equal-signal start bins on [1/K, 1], <= 30 %
background uncertainty, <= 15 bins; background categories: 5 equal-background start bins, <= 10 %, <= 5 bins; per bin
>= 1 background event, >= 1 unweighted event of every raw-count group, no negative template); or thesis, the custom
edges of the HH category and 20 equal bins elsewhere, merged by the AutoRebin(1.0, 0.9) of the thesis. Every HH
category is blinded from the first bin with S/B(mu=1) > 0.5 % (N14); the jet-fake estimate is clipped after the
rebinning (SH6).

The control variable NN_score (the score of the predicted class over every event) leaves out the events predicted as
the signal class in the nominal region (spec 6.4, decided 2026-10-25: only the signal category is blinded) and in the
regions that estimate it (nn_score_blinding).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from shapesmith.model import ABCD, NOMINAL, NOMINAL_REGION, Category, DataMinus, Estimator, Region, ScoreBinning, Variable
from shapesmith.nn_binning import BLINDING_S_OVER_B, THESIS_HH_EDGES, THESIS_OTHER_EDGES, category_rules

from bbtautau_shapesmith.constants import NN_CLASS_COLUMN, NN_SCORE_COLUMN

NN_FINE_BINNING = tuple(float(round(x, 6)) for x in np.linspace(0.0, 1.0, 1001))
NN_SIGNAL_CLASS = "HH2B2Tau"
NN_BTAG_SPLIT = {"btag1": "(n_bjets == 1)", "btag2": "(n_bjets == 2)"}  # K2
NN_NO_BTAG3P = "(n_bjets < 3)"  # K1, K2 (N12)


def nn_binning(classes: tuple[str, ...], signal: bool, method: str, required: Mapping[str, tuple[str, ...]], clip: tuple[str, ...]) -> ScoreBinning:
    """The binning of an NN category: the fine bins, the data-blind rules or the thesis start edges, the blinding of
    an HH category and the processes clipped after the rebinning."""
    blinding = BLINDING_S_OVER_B if signal else None
    if method == "thesis":
        return ScoreBinning(NN_FINE_BINNING, thesis=THESIS_HH_EDGES if signal else THESIS_OTHER_EDGES, blinding=blinding, clip=clip)
    return ScoreBinning(NN_FINE_BINNING, rules=category_rules(len(classes), signal, required), blinding=blinding, clip=clip)


def categories(classes: tuple[str, ...], scheme: str = "k1", binning: str = "data_blind", prefix: str = "", required: Mapping[str, tuple[str, ...]] | None = None,
               clip: tuple[str, ...] = ()) -> tuple[Category, ...]:
    """The NN categories (module docstring) of the classes in the order of predicted_class, from the friend columns of
    the model with the column prefix `prefix`; `required`: the raw-count groups of the data-blind rules."""
    result = []
    for index, name in enumerate(classes):
        cut = f"({prefix}{NN_CLASS_COLUMN} == {index})"
        signal = name == NN_SIGNAL_CLASS
        variable = Variable("NN_score", f"{prefix}{NN_SCORE_COLUMN}", nn_binning(classes, signal, binning, required or {}, clip))
        if scheme == "thesis":
            result.append(Category(name, cut, variable))
        elif scheme == "k2" and signal:
            result += [Category(f"{name}_{btag}", f"{cut} & {btags}", variable) for btag, btags in NN_BTAG_SPLIT.items()]
        else:
            result.append(Category(name, f"{cut} & {NN_NO_BTAG3P}", variable))
    return tuple(result)


def nn_score_blinding(classes: tuple[str, ...], prefix: str, cuts: Mapping[str, str], regions: Iterable[Region], estimators: Iterable[Estimator]) -> str:
    """The events that the control variable NN_score leaves out: those predicted as the signal class (the signal
    category before its n_bjets cut) in the nominal region and in the regions whose histograms are the shape of an
    estimate of the nominal region (the anti-isolated region of the fake factors, region b of ABCD), so that its
    jet-fake prediction holds the same events as its data and simulation. Per event: an event of any region passes
    every cut that no region replaces, so each of these regions is its version of the cuts that some region replaces
    (`cuts`: the nominal ones). A region that overlaps them loses those events too (with control_regions the os pass and
    fail regions of btag1, btag2 and btag3p); every other region (same sign, lepton anti-isolation, ...) keeps every
    event."""
    regions = tuple(regions)
    signal = f"({prefix}{NN_CLASS_COLUMN} == {classes.index(NN_SIGNAL_CLASS)})"
    replaced = {name for region in regions for name in region.replace_cuts}
    if not replaced:
        return signal
    by_name = {region.name: region for region in regions}
    shapes = [e.region if isinstance(e, DataMinus) else e.b for e in estimators if isinstance(e, (DataMinus, ABCD)) and e.target == NOMINAL]
    selections = []
    for region in (NOMINAL_REGION, *(by_name[name] for name in dict.fromkeys(shapes))):
        selected = {**cuts, **region.replace_cuts}
        selections.append(" & ".join(f"({expr})" for name, expr in selected.items() if name in replaced))
    return f"{signal} & ({' | '.join(f'({selection})' for selection in selections)})"


def blinded_score(score: str, blinded: str) -> str:
    """The score with the events of `blinded` moved below every axis (score - 2 <= -1, out of every histogram)."""
    return f"{score} - 2 * ({blinded})"


def nominal_category() -> tuple[Category, ...]:
    """Every event of the nominal region in one bin: the fit category of a channel without NN categories."""
    return (Category("nominal", "True", control_variables()["yield"]),)


def _edges(values) -> tuple[float, ...]:
    return tuple(float(round(v, 6)) for v in values)


_BINNING = {
    "pt_1": np.concatenate(([0], np.arange(20, 145, 5))),
    "pt_2": np.concatenate(([0], np.arange(20, 145, 5))),
    "eta_1": np.linspace(-2.5, 2.5, 51),
    "eta_2": np.linspace(-2.5, 2.5, 51),
    "phi_1": np.linspace(-np.pi, np.pi, 51),
    "phi_2": np.linspace(-np.pi, np.pi, 51),
    "iso_1": np.linspace(0.0, 1.0, 51),
    "iso_2": np.linspace(0.0, 1.0, 51),
    "mass_1": np.arange(0.0, 2.15, 0.1),
    "mass_2": np.arange(0.0, 2.15, 0.1),
    "tau_decaymode_1": np.arange(-0.5, 12.5, 1),
    "tau_decaymode_2": np.arange(-0.5, 12.5, 1),
    "q_1": [-4, 4],
    "m_vis": np.arange(0, 205, 5),
    "pt_vis": np.arange(0, 260, 10),
    "mt_1": np.arange(0, 165, 5),
    "mt_2": np.arange(0, 165, 5),
    "mt_tot": np.arange(0, 408, 8),
    "pt_tautau": np.arange(0, 205, 5),
    "deltaR_ditaupair": np.arange(0, 5.4, 0.2),
    "met": np.arange(0, 165, 5),
    "metphi": np.linspace(-np.pi, np.pi, 51),
    "metSumEt": np.arange(0, 710, 10),
    "pzetamissvis": np.arange(-350, 255, 5),
    "mTdileptonMET": np.arange(0, 204, 4),
    "n_jets": np.arange(-0.5, 8.5, 1),
    "n_bjets": np.arange(-0.5, 8.5, 1),
    "jpt_1": np.concatenate(([0], np.arange(20, 255, 5))),
    "jpt_2": np.concatenate(([0], np.arange(20, 255, 5))),
    "jeta_1": np.linspace(-3, 3, 31),
    "jeta_2": np.linspace(-3, 3, 31),
    "jphi_1": np.linspace(-np.pi, np.pi, 31),
    "jphi_2": np.linspace(-np.pi, np.pi, 31),
    "mjj": np.arange(0, 310, 10),
    "pt_dijet": np.arange(0, 205, 5),
    "jet_hemisphere": [-0.5, 0.5, 1.5],
    "bpair_pt_1": np.arange(0, 270, 10),
    "bpair_pt_2": np.arange(0, 190, 10),
    "bpair_eta_1": np.linspace(-3, 3, 21),
    "bpair_eta_2": np.linspace(-3, 3, 21),
    "bpair_phi_1": np.linspace(-np.pi, np.pi, 21),
    "bpair_phi_2": np.linspace(-np.pi, np.pi, 21),
    "bpair_btag_value_1": np.linspace(0.0, 1.0, 21),
    "bpair_btag_value_2": np.linspace(0.0, 1.0, 21),
    "bpair_m_inv": np.arange(0, 440, 20),
    "bpair_pt_dijet": np.arange(0, 320, 20),
    "bpair_deltaR": np.arange(0, 5.2, 0.2),
    "pt_tautaubb": np.arange(0, 520, 20),
    "mass_tautaubb": np.arange(100, 930, 30),
}
# switch diagnostics: the MET without recoil correction and jet/lepton propagation, the pileup proxies (v6 n-tuples)
_DIAGNOSTIC_BINNING = {
    "met_uncorrected": np.arange(0, 165, 5),
    "metphi_uncorrected": np.linspace(-np.pi, np.pi, 51),
    "npv_good": np.arange(-0.5, 80.5, 1),
    "rho": np.linspace(0, 60, 61),
}
DIAGNOSTIC_VARIABLES = tuple(_DIAGNOSTIC_BINNING)
# switch fastmtt_friend: the di-tau system of the FastMTT friend (et, mt, tt)
_FASTMTT_BINNING = {
    "m_fastmtt": np.arange(0, 410, 10),
    "pt_fastmtt": np.arange(0, 310, 10),
    "eta_fastmtt": np.linspace(-5, 5, 51),
    "phi_fastmtt": np.linspace(-np.pi, np.pi, 51),
}
FASTMTT_VARIABLES = tuple(_FASTMTT_BINNING)
_EXPRESSIONS = {"yield": "1.0", "sum_deltaR_tt_bb": "bpair_deltaR + deltaR_ditaupair", "NN_score": NN_SCORE_COLUMN}
_EXPRESSION_BINNING = {"yield": (0.5, 1.5), "sum_deltaR_tt_bb": np.arange(0, 10.6, 0.2), "NN_score": np.linspace(0.0, 1.0, 21)}


def control_variables(nn_friend: bool = False, diagnostics: bool = False, fastmtt: bool = False, nn_prefix: str = "", nn_blinded: str | None = None) -> dict[str, Variable]:
    """Control variables (`yield` is the one-bin event count); the NN score (of the model with the column prefix
    nn_prefix, without the events of nn_blinded: nn_score_blinding) only when the NN friend tree is part of the run
    (nn_friend), the diagnostic variables only with diagnostics, the FastMTT variables only with its friend."""
    binning = {**_BINNING, **(_DIAGNOSTIC_BINNING if diagnostics else {}), **(_FASTMTT_BINNING if fastmtt else {})}
    variables = {name: Variable(name, name, _edges(edges)) for name, edges in binning.items()}
    for name, expr in _EXPRESSIONS.items():
        if name == "NN_score" and not nn_friend:
            continue
        if name == "NN_score":
            expr = f"{nn_prefix}{expr}" if nn_blinded is None else blinded_score(f"{nn_prefix}{expr}", nn_blinded)
        variables[name] = Variable(name, expr, _edges(_EXPRESSION_BINNING[name]))
    return variables
