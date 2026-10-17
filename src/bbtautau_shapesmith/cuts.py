"""Tau channels: baseline and skim selection per era, gen-match splits, estimation and control regions (v15 branch
names). The trigger of the era and channel comes from triggers.TRIGGER (v7 spec 6.2); every era cuts the light lepton
of et and mt to |eta| < 2.1 (E31) and both taus of tt to |eta| < 2.1 (E7, E33: the embedding has no trigger matching),
2018 vetoes the HEM 15/16 events (E29)."""
from __future__ import annotations

from shapesmith.model import Region

from bbtautau_shapesmith.constants import (
    BTAG_BINS, HEM_ERAS, HEM_VETO, LEPTON_ETA, LT_CHANNELS, TAU_LEGS, TAU_VS_ELE_WP, TAU_VS_JET_LOOSE_WP, TAU_VS_JET_WP, TAU_VS_MU_WP,
)
from bbtautau_shapesmith.triggers import TRIGGER
from bbtautau_shapesmith.weights import fail_tau_weights, fake_factor_weight

SAME_SIGN = "((q_1 * q_2) > 0)"
TT_TAU_ETA = "(abs(eta_1) < 2.1) & (abs(eta_2) < 2.1)"
B_TAGGING = "(n_bjets >= 1) & (bpair_pt_2 > 0)"  # a b pair with at least one b tag


def isolated(leg: int) -> str:
    return f"(id_tau_vsJet_{TAU_VS_JET_WP}_{leg} > 0.5)"


def loose(leg: int) -> str:
    return f"(id_tau_vsJet_{TAU_VS_JET_LOOSE_WP}_{leg} > 0.5)"


def anti_isolated(leg: int) -> str:
    return f"((id_tau_vsJet_{TAU_VS_JET_WP}_{leg} < 0.5) & (id_tau_vsJet_{TAU_VS_JET_LOOSE_WP}_{leg} > 0.5))"


def era_cuts(era: str) -> dict[str, str]:
    """The cuts of an era on every channel: the HEM veto in 2018."""
    return {"hem_veto": HEM_VETO} if era in HEM_ERAS else {}


def baseline_cuts(channel: str, era: str) -> dict[str, str]:
    cuts = {
        "extraelec_veto": "(extraelec_veto < 0.5)",
        "extramuon_veto": "(extramuon_veto < 0.5)",
        "os": "((q_1 * q_2) < 0)",
        "b_tagging": B_TAGGING,
        "against_muon": " & ".join(f"(id_tau_vsMu_{TAU_VS_MU_WP[channel]}_{leg} > 0.5)" for leg in TAU_LEGS[channel]),
        "against_electron": " & ".join(f"(id_tau_vsEle_{TAU_VS_ELE_WP[channel]}_{leg} > 0.5)" for leg in TAU_LEGS[channel]),
        "tau_iso": " & ".join(isolated(leg) for leg in TAU_LEGS[channel]),
    }
    if channel in LT_CHANNELS:
        cuts["dilepton_veto"] = "(dilepton_veto < 0.5)"  # not produced for tt (no light lepton in the final state)
        cuts["lepton_iso"] = "(iso_1 < 0.15)"
        cuts["mt_cut"] = "(mt_1 < 70)"
        # EMB has hardly any tau_h beyond 2.3, where the Run-2 vsMu SFs end
        cuts["tau_eta"] = "(abs(eta_2) < 2.3)"
        cuts["lepton_eta"] = LEPTON_ETA
    else:
        cuts["pt_selection"] = "(pt_1 > 40) & (pt_2 > 40)"
        # the acceptance of the di-tau trigger (E7): data and MC have it from the trigger matching, the untriggered
        # embedding (E33) needs the cut, as its eps_data is measured for |eta| < 2.1
        cuts["tau_eta"] = TT_TAU_ETA
    cuts["trg_selection"] = TRIGGER[era][channel].cut
    return {**cuts, **era_cuts(era)}


def skim_cuts(channel: str, era: str, control_regions: bool = False) -> dict[str, str]:
    """Baseline without the charge requirement and with the loosest tau isolation, so that all estimation regions survive."""
    baseline = baseline_cuts(channel, era)
    cuts = {name: expr for name, expr in baseline.items() if name != "os"}
    cuts["tau_iso"] = " & ".join(loose(leg) for leg in TAU_LEGS[channel])
    if control_regions:
        # >= 2 jets for the b-tag bins, and the nominal acceptance too: n_bjets and n_jets can have different pT thresholds.
        cuts["b_tagging"] = f"(n_jets >= 2) | ({baseline['b_tagging']})"
        if channel in LT_CHANNELS:
            cuts.pop("mt_cut")
            cuts["lepton_iso"] = "iso_1 < 0.5"
    return cuts


def genmatch_cuts(channel: str) -> dict[str, str]:
    """T = genuine di-tau (the lepton leg from a tau decay), J = jet -> tau_h fake, L = everything else (lepton fakes)."""
    true_tau = {
        "mt": "((gen_match_1 == 4) & (gen_match_2 == 5))",
        "et": "((gen_match_1 == 3) & (gen_match_2 == 5))",
        "tt": "((gen_match_1 == 5) & (gen_match_2 == 5))",
    }[channel]
    jet_fake = "((gen_match_1 == 6) | (gen_match_2 == 6))" if channel == "tt" else "(gen_match_2 == 6)"
    return {"T": true_tau, "L": f"(~{true_tau} & ~{jet_fake})", "J": jet_fake}


def anti_iso_cut(channel: str) -> str:
    if channel in LT_CHANNELS:
        return anti_isolated(2)
    return f"(({isolated(1)} & {anti_isolated(2)}) | ({anti_isolated(1)} & {isolated(2)}))"


def regions(channel: str, jet_fakes: str, ff_type: str = "corrected") -> tuple[Region, ...]:
    """Estimation regions: same-sign always; the fake-factor region (FF friend weight of ff_type) in ff mode, the ABCD regions in mc mode.

    Genuine taus keep the Medium pass SF in the anti-isolated regions (TauFakeFactors and legacy precedent)."""
    anti = anti_iso_cut(channel)
    same_sign = Region("same_sign", replace_cuts={"os": SAME_SIGN})
    if jet_fakes == "ff":
        return (same_sign, Region("anti_iso", replace_cuts={"tau_iso": anti}, add_weights={"fake_factor": fake_factor_weight(channel, ff_type)}))
    return (
        same_sign,
        Region("abcd_same_sign", replace_cuts={"os": SAME_SIGN}),
        Region("abcd_anti_iso", replace_cuts={"tau_iso": anti}),
        Region("abcd_same_sign_anti_iso", replace_cuts={"os": SAME_SIGN, "tau_iso": anti}),
    )


def _pass_fail(name: str, cuts: dict[str, str], channel: str) -> tuple[Region, ...]:
    """The pass region and the tau-fail region, whose genuine taus lose the pass SF (MC only: EMB has its own weights)."""
    return (Region(f"{name}_pass", replace_cuts=cuts),
            Region(f"{name}_fail", replace_cuts={**cuts, "tau_iso": anti_iso_cut(channel)}, replace_weights=fail_tau_weights(channel)))


def diagnostic_regions(channel: str) -> tuple[Region, ...]:
    """Raw pass/fail controls, independent of the nominal ABCD/FF estimate."""
    result = ()
    for bin_name, btags in BTAG_BINS.items():
        jets = f"({btags}) & (n_jets >= 2)"
        for charge, charge_cut in (("os", "(q_1 * q_2) < 0"), ("ss", SAME_SIGN)):
            result += _pass_fail(f"{bin_name}_{charge}", {"os": charge_cut, "b_tagging": jets}, channel)
    if channel in LT_CHANNELS:
        result += _pass_fail("w_highmt", {"mt_cut": "mt_1 > 80", "b_tagging": "(n_bjets == 0) & (n_jets >= 2)"}, channel)
        for charge, charge_cut in (("", "(q_1 * q_2) < 0"), ("_ss", SAME_SIGN)):
            # the MC muon iso SF covers isolated muons only; EMB keeps its iso-binned SF
            result += (Region(f"lepton_antiiso{charge}",
                              replace_cuts={"lepton_iso": "(iso_1 >= 0.15) & (iso_1 < 0.5)", "os": charge_cut},
                              replace_weights={"iso": "1.0"} if channel == "mt" else {}),)
    return result
