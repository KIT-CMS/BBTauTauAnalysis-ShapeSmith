"""The triggers per era and channel (v7 spec 6.2): TRIGGER[era][channel], with the offline thresholds and the trigger
weights of the simulated and the embedded samples.

Flags and weights are the CROWN columns of the v7 configuration (spec C6-C8, E2-E7, E33):
- muon (mt, em, mm, the tau-ID measurement): 2016 the OR of IsoMu22, IsoTkMu22, IsoMu22_eta2p1 and IsoTkMu22_eta2p1,
  pT > 24 GeV and |eta| < 2.1 for the triggering muon (E2); 2017 IsoMu27, pT > 29 GeV (E3); 2018 IsoMu24 (E4). The scale
  factor of the era (E5): trg_wgt_single_mu22 / _mu27 / _mu24, on MC the POG muon_Z SF (2016 IsoMu24_or_IsoTkMu24, 2017
  IsoMu27, 2018 IsoMu24; productions smrun2_binned_v7b and smrun2_tau_id_v2 on), on the embedding the KIT SF; the column
  names stay. smrun2_binned_v7 and smrun2_tau_id_v1 carry the KIT SF on MC as well.
- electron (et, ee): 2016 Ele25_eta2p1, pT > 27 GeV (|eta| < 2.1); 2017 Ele32 for 34 <= pT < 37 GeV and Ele35 from 37
  GeV, the weight piecewise and, above 37 GeV, the EGM HLT Zvtx factor 0.991 on MC and embedding (E6, D4); 2018 Ele32.
- di-tau (tt, E7): 2016 the data take DoubleMediumIsoPFTau35 or DoubleMediumCombinedIsoPFTau35, the MC the Iso path only
  (D9); 2017 the three 2017 paths; 2018 the four 2018 paths. MC: the leg product of the POG di-tau SF of the first fired
  path (CROWN columns trg_wgt<flag without "trg">_leg1/2). Embedding (E33, variant A): no trigger requirement, the
  weight eps_data(tau1) * eps_data(tau2), each leg in its isolation class (Medium, or VVVLoose and not Medium).
The tt cut depends on the sample kind through the skim columns is_data, is_mc and is_embedding.

The light-lepton |eta| < 2.1 of et and mt in all eras (E31) is the cut constants.LEPTON_ETA of the tau channels, the
tau-ID measurement's own cut; here only the 2016 dilepton channels carry the trigger's eta limit.
"""
from __future__ import annotations

from dataclasses import dataclass

from bbtautau_shapesmith.constants import COMBINED_ERAS, ERAS, TAU_VS_JET_WP, year


@dataclass(frozen=True)
class Trigger:
    cut: str  # the trigger requirement (flags, and in the tau channels the offline thresholds)
    mc_weight: str  # trigger scale factor of the simulated samples
    embedding_weight: str  # trigger weight of the embedded samples
    zvtx: str | None = None  # 2017 Ele35: the HLT Zvtx factor on MC and embedding (E6)
    thresholds: str | None = None  # the offline pT thresholds of a dilepton channel (its cut pt_selection)


MUON_FLAGS = {
    "2016": ("trg_single_mu22", "trg_single_mu22_tk", "trg_single_mu22_eta2p1", "trg_single_mu22_eta2p1_tk"),
    "2017": ("trg_single_mu27",),
    "2018": ("trg_single_mu24",),
}
MUON_WEIGHT = {"2016": "trg_wgt_single_mu22", "2017": "trg_wgt_single_mu27", "2018": "trg_wgt_single_mu24"}
# the flags of the di-tau paths: data, MC (E7: in 2016 the MC takes the Iso path only, D9)
DITAU_FLAGS = {
    "2016": {"data": ("trg_double_tau35_mediumiso", "trg_double_tau35_mediumcombinediso"), "mc": ("trg_double_tau35_mediumiso",)},
    "2017": {kind: ("trg_double_tau35_tightiso", "trg_double_tau40_mediumiso", "trg_double_tau40_tightiso") for kind in ("data", "mc")},
    "2018": {kind: ("trg_double_tau35_mediumiso", "trg_double_tau35_tightiso", "trg_double_tau40_mediumiso", "trg_double_tau40_tightiso") for kind in ("data", "mc")},
}
ZVTX = "((pt_1 >= 37) * 0.991 + (pt_1 < 37) * 1.0)"  # 2017: EGM HLT Zvtx SF of the Ele35 events (D4), 1.0 for Ele32
EMBEDDING_DITAU_EFFICIENCY = {"Medium": "trg_effdata_emb_Medium_{leg}", "AntiMedium": "trg_effdata_emb_AntiMedium_{leg}"}  # CROWN, E33


def fired(flags: tuple[str, ...]) -> str:
    """At least one of the flags; a single flag without extra parentheses."""
    terms = [f"({flag} > 0.5)" for flag in flags]
    return terms[0] if len(terms) == 1 else "(" + " | ".join(terms) + ")"


def ditau_sf(flag: str, leg: int) -> str:
    """CROWN's POG di-tau SF column of a flag and leg: trg_<path> -> trg_wgt<path>_leg<n> (no underscore after wgt)."""
    return f"{flag.replace('trg_', 'trg_wgt', 1)}_leg{leg}"


def ditau_mc_weight(flags: tuple[str, ...]) -> str:
    """Leg product of the SF of the first fired path, in the given priority."""
    terms = []
    for index, flag in enumerate(flags):
        conditions = [f"({earlier} < 0.5)" for earlier in flags[:index]] + [f"({flag} > 0.5)"]
        # booleans are combined with & and multiplied into the weight as one factor (numexpr has no bool * bool)
        terms.append("(" + " & ".join(conditions) + f") * ({ditau_sf(flag, 1)} * {ditau_sf(flag, 2)})")
    return "(" + " + ".join(terms) + ")"


def ditau_embedding_weight() -> str:
    """E33: eps_data of both taus, each in its isolation class: Medium where it passes the analysis WP, else the
    anti-isolated class VVVLoose and not Medium (the AR of the fake-factor method)."""
    legs = []
    for leg in (1, 2):
        iso = f"(id_tau_vsJet_{TAU_VS_JET_WP}_{leg} > 0.5)"
        anti = f"(id_tau_vsJet_{TAU_VS_JET_WP}_{leg} < 0.5)"
        legs.append(f"({iso} * {EMBEDDING_DITAU_EFFICIENCY['Medium'].format(leg=leg)} + {anti} * {EMBEDDING_DITAU_EFFICIENCY['AntiMedium'].format(leg=leg)})")
    return " * ".join(legs)


def ditau_cut(year: str) -> str:
    """Embedding without trigger requirement (E33); data and MC with their flags (2016: data OR, MC Iso only)."""
    flags = DITAU_FLAGS[year]
    if flags["data"] == flags["mc"]:
        return f"((is_embedding > 0.5) | {fired(flags['data'])})"
    return f"((is_embedding > 0.5) | ((is_data > 0.5) & {fired(flags['data'])}) | ((is_mc > 0.5) & {fired(flags['mc'])}))"


def _single(cut: str, weight: str, zvtx: str | None = None, thresholds: str | None = None) -> Trigger:
    return Trigger(cut, weight, weight, zvtx, thresholds)


ELECTRON_2017 = "(((pt_1 >= 34) & (pt_1 < 37) & (trg_single_ele32 > 0.5)) | ((pt_1 >= 37) & (trg_single_ele35 > 0.5)))"
ELECTRON_2017_WEIGHT = "((pt_1 < 37) * trg_wgt_single_ele32 + (pt_1 >= 37) * trg_wgt_single_ele35)"


def _year_table(year: str) -> dict[str, Trigger]:
    muon, muon_weight = fired(MUON_FLAGS[year]), MUON_WEIGHT[year]
    ditau = Trigger(ditau_cut(year), ditau_mc_weight(DITAU_FLAGS[year]["mc"]), ditau_embedding_weight())
    if year == "2016":
        return {
            "mt": _single(f"(pt_2 > 30) & (pt_1 > 24) & {muon}", muon_weight),
            "et": _single("(pt_2 > 30) & (pt_1 > 27) & (trg_single_ele25 > 0.5)", "trg_wgt_single_ele25"),
            "tt": ditau,
            # the triggering muon (mm: leg 1, em: leg 2) and electron (ee: leg 1) within the |eta| < 2.1 of the trigger
            "em": _single(f"(abs(eta_2) < 2.1) & {muon}", muon_weight, thresholds="(pt_1 > 15) & (pt_2 > 24)"),
            "mm": _single(f"(abs(eta_1) < 2.1) & {muon}", muon_weight, thresholds="(pt_1 > 24) & (pt_2 > 15)"),
            "ee": _single("(abs(eta_1) < 2.1) & (trg_single_ele25 > 0.5)", "trg_wgt_single_ele25", thresholds="(pt_1 > 27) & (pt_2 > 15)"),
            "tau_id": _single(f"(pt_1 >= 24) & {muon}", muon_weight),
        }
    if year == "2017":
        return {
            "mt": _single(f"(pt_2 > 30) & (pt_1 > 29) & {muon}", muon_weight),
            "et": _single(f"(pt_2 > 30) & {ELECTRON_2017}", ELECTRON_2017_WEIGHT, ZVTX),
            "tt": ditau,
            "em": _single(muon, muon_weight, thresholds="(pt_1 > 15) & (pt_2 > 29)"),
            "mm": _single(muon, muon_weight, thresholds="(pt_1 > 29) & (pt_2 > 15)"),
            "ee": _single(ELECTRON_2017, ELECTRON_2017_WEIGHT, ZVTX, thresholds="(pt_1 >= 34) & (pt_2 > 15)"),
            "tau_id": _single(f"(pt_1 >= 29) & {muon}", muon_weight),
        }
    return {  # 2018: the v6 selections; the dilepton thresholds are the effective ones of CROWN's flags
        "mt": _single(f"(pt_2 > 30) & (pt_1 > 25) & {muon}", muon_weight),
        "et": _single("(pt_2 > 30) & (pt_1 > 33) & (trg_single_ele32 > 0.5)", "trg_wgt_single_ele32"),
        "tt": ditau,
        "em": _single("trg_single_mu24 > 0.5", muon_weight, thresholds="(pt_1 > 15) & (pt_2 > 26)"),
        "mm": _single("trg_single_mu24 > 0.5", muon_weight, thresholds="(pt_1 > 26) & (pt_2 > 15)"),
        "ee": _single("trg_single_ele32 > 0.5", "trg_wgt_single_ele32", thresholds="(pt_1 > 34) & (pt_2 > 15)"),
        # the tau-ID measurement takes IsoMu24 or IsoMu27, weighted as its predecessor (smhtt_ul process_selection.py,
        # with the gap at pt_1 == 28 exactly)
        "tau_id": _single(
            "(pt_1 >= 25) & ((trg_single_mu27 > 0.5) | (trg_single_mu24 > 0.5))",
            "((pt_1 >= 25) & (pt_1 < 28)) * trg_wgt_single_mu24 + (pt_1 > 28) * trg_wgt_single_mu27",
        ),
    }


TRIGGER = {era: _year_table(year(era)) for era in (*ERAS, *COMBINED_ERAS)}  # a combined era (2016) has the table of its year
