"""Systematic uncertainties and plot style: b-tag weight variations (shapes), normalisation uncertainties (lnN),
colours, legend labels, stack order and axis labels (ported from Dumbledraw's non_res_hh branch, mathtext)."""
from __future__ import annotations

from shapesmith.model import LnN, Process, Style, WeightVariation

from bbtautau_shapesmith.constants import BTAG_COMPONENTS, CHANNELS, ERA
from bbtautau_shapesmith.processes import SIGNAL

LT = ("et", "mt")
TT = ("tt",)


def weight_variations() -> tuple[WeightVariation, ...]:
    return tuple(
        WeightVariation(f"CMS_btag_{component}_{ERA}{shift}", {"btag": f"btag_weight_upart_{shift.lower()}_{component}"})
        for component in BTAG_COMPONENTS
        for shift in ("Up", "Down")
    )


def lnn(processes: tuple[Process, ...], jet_fakes_output: str) -> tuple[LnN, ...]:
    mc = tuple(p.name for p in processes if p.kind not in ("data", "embedding"))

    def group(*groups: str) -> tuple[str, ...]:
        return tuple(p.name for p in processes if p.group in groups)

    entries = [
        LnN("lumi_13TeV_$ERA", mc, 1.025),
        LnN("eff_e", mc, 1.02, channels=("et",)),
        LnN("eff_m", mc, 1.02, channels=("mt",)),
        LnN("eff_t_vsLep_$CHANNEL_$ERA", mc, 1.01, channels=LT),
        LnN("eff_t_vsLep_$CHANNEL_$ERA", mc, 1.0201, channels=TT),
        LnN("eff_t_vsJet_$CHANNEL_$ERA", mc, 1.02, channels=LT),
        LnN("eff_t_vsJet_$CHANNEL_$ERA", mc, 1.0404, channels=TT),
        LnN("htt_zjXsec", group("DY"), 1.02),
        LnN("htt_tjXsec", group("TT"), 1.06),
        LnN("htt_stXsec", group("ST"), 1.05),
        LnN("htt_vvXsec", group("VV"), 1.05),
        LnN("htt_ttvXsec", group("TTV"), 1.10),
        LnN("htt_ewkXsec", group("EWK"), 1.05),
        LnN("htt_wjXsec", group("W"), 1.04),
        LnN("QCDscale_ggH", group("ggH"), 1.039), LnN("pdf_ggH", group("ggH"), 1.019), LnN("alphaS_ggH", group("ggH"), 1.026),
        LnN("QCDscale_qqH", group("qqH"), (0.997, 1.004)), LnN("pdf_qqH", group("qqH"), 1.021), LnN("alphaS_qqH", group("qqH"), 1.005),
        LnN("QCDscale_ttH", group("ttH"), (0.908, 1.058)), LnN("pdf_ttH", group("ttH"), 1.030), LnN("alphaS_ttH", group("ttH"), 1.020),
        LnN("QCDscale_VH", group("VH"), (0.970, 1.038)), LnN("pdf_VH", group("VH"), 1.017), LnN("alphaS_VH", group("VH"), 1.009),
        LnN("QCDscale_HH", (SIGNAL,), (0.95, 1.022)), LnN("PDF_alphas_HH", (SIGNAL,), 1.03), LnN("mtop_HH", (SIGNAL,), 1.026),
        LnN("BR_h_bb", (SIGNAL,), (0.9873, 1.0125)), LnN("BR_h_tautau", (SIGNAL,), 1.0165),
        LnN(f"{jet_fakes_output}Norm_$CHANNEL_$ERA", (jet_fakes_output,), 1.108, channels=LT),
        LnN(f"{jet_fakes_output}Norm_$CHANNEL_$ERA", (jet_fakes_output,), 1.216, channels=TT),
    ]
    return tuple(e for e in entries if e.processes)


COLORS = {
    "Z": "#3f90da", "TT": "#832db6", "ST": "#717581", "VV": "#94a4a2", "rare": "#009333",
    "jetFakes": "#b9ac70", "QCD": "#b9ac70", "EMB": "#ffa90e", "HH2B2Tau": "#bd1f01",
}
LABELS = {
    "Z": r"Z$\rightarrow\ell\ell$ / $\tau\tau$", "TT": r"t$\bar{\mathrm{t}}$", "ST": "Single t", "VV": "Diboson", "rare": "Other",
    "jetFakes": r"jet$\rightarrow\tau_h$", "QCD": "QCD multijet", "EMB": r"$\tau$ embedded",
}
GROUP_ORDER = ("rare", "QCD", "jetFakes", "EMB", "VV", "ST", "TT", "Z")
CHANNEL_LABELS = {"et": r"e$\tau_h$", "mt": r"$\mu\tau_h$", "tt": r"$\tau_h\tau_h$"}
LEG1 = {"et": "Electron", "mt": "Muon", "tt": r"Leading $\tau_h$"}
LEG2 = {"et": r"$\tau_h$", "mt": r"$\tau_h$", "tt": r"Trailing $\tau_h$"}

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
    "iso_2": r"$\tau_h$ isolation", "mass_2": r"$\tau_h$ mass / GeV", "tau_decaymode_1": "Decay mode (leg 1)", "tau_decaymode_2": r"$\tau_h$ decay mode",
    "phi_1": r"$\varphi$ (leg 1)", "phi_2": r"$\tau_h$ $\varphi$", "mass_1": "Mass (leg 1) / GeV", "q_1": "Charge (leg 1)", "NN_score": "NN output",
}


def axis_labels() -> dict[str, dict[str, str]]:
    labels = {}
    for channel in CHANNELS:
        labels[channel] = {
            **_COMMON_AXIS_LABELS,
            "pt_1": f"{LEG1[channel]} $p_T$ / GeV",
            "pt_2": f"{LEG2[channel]} $p_T$ / GeV",
            "eta_1": rf"{LEG1[channel]} $\eta$",
            "eta_2": rf"{LEG2[channel]} $\eta$",
            "iso_1": f"{LEG1[channel]} isolation",
            "mt_1": rf"$m_T$({LEG1[channel]}, $p_T^{{miss}}$) / GeV",
        }
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
