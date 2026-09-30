"""The NN categories and the control variables with their binning."""
from __future__ import annotations

import numpy as np
from shapesmith.model import Category, Variable

from bbtautau_shapesmith.constants import NN_CLASS_COLUMN, NN_CLASS_NAMES, NN_SCORE_COLUMN

STANDARD_BINNING = tuple(float(round(x, 4)) for x in np.linspace(0.0, 1.0, 21))
SIGNAL_BINNING = (0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0)


def categories() -> tuple[Category, ...]:
    return tuple(
        Category(name, f"({NN_CLASS_COLUMN} == {index})", Variable("NN_score", NN_SCORE_COLUMN, SIGNAL_BINNING if name == "HH2B2Tau" else STANDARD_BINNING))
        for index, name in enumerate(NN_CLASS_NAMES)
    )


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
_EXPRESSIONS = {"yield": "1.0", "sum_deltaR_tt_bb": "bpair_deltaR + deltaR_ditaupair", "NN_score": NN_SCORE_COLUMN}
_EXPRESSION_BINNING = {"yield": (0.5, 1.5), "sum_deltaR_tt_bb": np.arange(0, 10.6, 0.2), "NN_score": np.linspace(0.0, 1.0, 21)}


def control_variables(nn_friend: bool = False) -> dict[str, Variable]:
    """Control variables (`yield` is the one-bin event count); the NN score only when the NN friend tree is part of the run (nn_friend)."""
    variables = {name: Variable(name, name, _edges(edges)) for name, edges in _BINNING.items()}
    for name, expr in _EXPRESSIONS.items():
        if name == "NN_score" and not nn_friend:
            continue
        variables[name] = Variable(name, expr, _edges(_EXPRESSION_BINNING[name]))
    return variables
