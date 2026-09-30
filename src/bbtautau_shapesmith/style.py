"""Plot style: colours, legend labels, stack order and axis titles of all six channels (ported from Dumbledraw's
non_res_hh branch, mathtext)."""
from __future__ import annotations

from shapesmith.model import Style

from bbtautau_shapesmith.constants import DILEPTON_CHANNELS, TAU_CHANNELS

COLORS = {
    "Z": "#3f90da", "TT": "#832db6", "ST": "#717581", "VV": "#94a4a2", "rare": "#009333",
    "jetFakes": "#b9ac70", "QCD": "#b9ac70", "EMB": "#ffa90e", "HH2B2Tau": "#bd1f01",
}
LABELS = {
    "Z": r"Z$\rightarrow\ell\ell$ / $\tau\tau$", "TT": r"t$\bar{\mathrm{t}}$", "ST": "Single t", "VV": "Diboson", "rare": "Other",
    "jetFakes": r"jet$\rightarrow\tau_h$", "QCD": "QCD multijet", "EMB": r"$\tau$ embedded",
}
GROUP_ORDER = ("rare", "QCD", "jetFakes", "EMB", "VV", "ST", "TT", "Z")
CHANNEL_LABELS = {"et": r"e$\tau_h$", "mt": r"$\mu\tau_h$", "tt": r"$\tau_h\tau_h$", "em": r"e$\mu$", "mm": r"$\mu\mu$", "ee": "ee"}
LEG1 = {"et": "Electron", "mt": "Muon", "tt": r"Leading $\tau_h$", "em": "Electron", "mm": "Leading muon", "ee": "Leading electron"}
LEG2 = {"et": r"$\tau_h$", "mt": r"$\tau_h$", "tt": r"Trailing $\tau_h$", "em": "Muon", "mm": "Trailing muon", "ee": "Trailing electron"}

_COMMON_AXIS_LABELS = {
    "m_vis": r"$m_{vis}$ / GeV", "pt_vis": r"$p_T^{vis}$ / GeV", "mt_tot": r"$m_T^{tot}$ / GeV", "pt_tautau": r"$p_T(\tau\tau)$ / GeV",
    "deltaR_ditaupair": r"$\Delta R(\tau\tau)$", "met": r"$p_T^{miss}$ / GeV", "metphi": r"$\varphi(p_T^{miss})$", "metSumEt": r"$\sum E_T$ / GeV",
    "pzetamissvis": r"$D_\zeta$ / GeV", "mTdileptonMET": r"$m_T(\tau\tau, p_T^{miss})$ / GeV", "mt_2": r"$m_T(\tau_h, p_T^{miss})$ / GeV",
    "n_jets": "Number of jets", "n_bjets": "Number of b-tagged jets", "jpt_1": r"Leading jet $p_T$ / GeV", "jpt_2": r"Trailing jet $p_T$ / GeV",
    "jeta_1": r"Leading jet $\eta$", "jeta_2": r"Trailing jet $\eta$", "jphi_1": r"Leading jet $\varphi$", "jphi_2": r"Trailing jet $\varphi$",
    "mjj": r"$m_{jj}$ / GeV", "pt_dijet": r"$p_T(jj)$ / GeV", "jet_hemisphere": "Jet hemisphere",
    "bpair_pt_1": r"Leading b-jet $p_T$ / GeV", "bpair_pt_2": r"Trailing b-jet $p_T$ / GeV", "bpair_eta_1": r"Leading b-jet $\eta$", "bpair_eta_2": r"Trailing b-jet $\eta$",
    "bpair_phi_1": r"Leading b-jet $\varphi$", "bpair_phi_2": r"Trailing b-jet $\varphi$", "bpair_btag_value_1": "Leading b-jet UParT score", "bpair_btag_value_2": "Trailing b-jet UParT score",
    "bpair_m_inv": r"$m_{bb}$ / GeV", "bpair_pt_dijet": r"$p_T(bb)$ / GeV", "bpair_deltaR": r"$\Delta R(bb)$",
    "pt_tautaubb": r"$p_T(bb\tau\tau)$ / GeV", "mass_tautaubb": r"$m(bb\tau\tau)$ / GeV", "sum_deltaR_tt_bb": r"$\Delta R(\tau\tau) + \Delta R(bb)$",
    "mass_2": r"$\tau_h$ mass / GeV", "tau_decaymode_1": "Decay mode (leg 1)", "tau_decaymode_2": r"$\tau_h$ decay mode",
    "mass_1": "Mass (leg 1) / GeV", "q_1": "Charge (leg 1)", "NN_score": "NN output", "yield": "Events",
}


def axis_labels() -> dict[str, dict[str, str]]:
    """Axis titles per channel: the common table plus the leg-dependent ones, Z-pair titles in the dilepton channels."""
    labels = {}
    for channel in TAU_CHANNELS + DILEPTON_CHANNELS:
        labels[channel] = {
            **_COMMON_AXIS_LABELS,
            "pt_1": f"{LEG1[channel]} $p_T$ / GeV", "pt_2": f"{LEG2[channel]} $p_T$ / GeV",
            "eta_1": rf"{LEG1[channel]} $\eta$", "eta_2": rf"{LEG2[channel]} $\eta$",
            "phi_1": rf"{LEG1[channel]} $\varphi$", "phi_2": rf"{LEG2[channel]} $\varphi$",
            "iso_1": f"{LEG1[channel]} isolation", "iso_2": f"{LEG2[channel]} isolation",
            "mt_1": rf"$m_T$({LEG1[channel]}, $p_T^{{miss}}$) / GeV",
        }
        if channel in DILEPTON_CHANNELS:
            labels[channel].update(m_vis=r"$m_{\ell\ell}$ / GeV", pt_vis=r"$p_T^{\ell\ell}$ / GeV")
    return labels


def style() -> Style:
    return Style(
        colors=COLORS,
        labels=LABELS,
        group_order=GROUP_ORDER,
        signal_label=r"HH$\rightarrow$bb$\tau\tau$",
        lumi_label=r"59.8 fb$^{-1}$ (2018, 13 TeV)",
        channel_labels=CHANNEL_LABELS,
        axis_labels=axis_labels(),
    )
