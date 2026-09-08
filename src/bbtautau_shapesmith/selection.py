"""Baseline selection, skim selection, estimation regions and MC weights per channel (v15 branch names)."""
from __future__ import annotations

from shapesmith.model import Channel, Region, Selection

from bbtautau_shapesmith.constants import FF_COLUMNS, LT_CHANNELS, TAU_VS_ELE_WP, TAU_VS_JET_LOOSE_WP, TAU_VS_JET_WP, TAU_VS_MU_WP

TRIGGER = {
    "mt": "(pt_2 > 30) & (pt_1 > 25) & (trg_single_mu24 > 0.5)",
    "et": "(pt_2 > 30) & (pt_1 > 33) & (trg_single_ele32 > 0.5)",
    "tt": "(trg_double_tau35_mediumiso > 0.5) | (trg_double_tau35_tightiso > 0.5) | (trg_double_tau40_mediumiso > 0.5) | (trg_double_tau40_tightiso > 0.5)",
}
SAME_SIGN = "((q_1 * q_2) > 0)"


def legs(channel: str) -> tuple[int, ...]:
    return (1, 2) if channel == "tt" else (2,)


def isolated(leg: int) -> str:
    return f"(id_tau_vsJet_{TAU_VS_JET_WP}_{leg} > 0.5)"


def loose(leg: int) -> str:
    return f"(id_tau_vsJet_{TAU_VS_JET_LOOSE_WP}_{leg} > 0.5)"


def anti_isolated(leg: int) -> str:
    return f"((id_tau_vsJet_{TAU_VS_JET_WP}_{leg} < 0.5) & (id_tau_vsJet_{TAU_VS_JET_LOOSE_WP}_{leg} > 0.5))"


def baseline_cuts(channel: str) -> dict[str, str]:
    cuts = {
        "extraelec_veto": "(extraelec_veto < 0.5)",
        "extramuon_veto": "(extramuon_veto < 0.5)",
        "os": "((q_1 * q_2) < 0)",
        "b_tagging": "(n_bjets >= 1) & (bpair_pt_2 > 0)",
        "against_muon": " & ".join(f"(id_tau_vsMu_{TAU_VS_MU_WP[channel]}_{leg} > 0.5)" for leg in legs(channel)),
        "against_electron": " & ".join(f"(id_tau_vsEle_{TAU_VS_ELE_WP[channel]}_{leg} > 0.5)" for leg in legs(channel)),
        "tau_iso": " & ".join(isolated(leg) for leg in legs(channel)),
    }
    if channel in LT_CHANNELS:
        cuts["dilepton_veto"] = "(dilepton_veto < 0.5)"  # not produced for tt (no light lepton in the final state)
        cuts["lepton_iso"] = "(iso_1 < 0.15)"
        cuts["mt_cut"] = "(mt_1 < 70)"
    else:
        cuts["pt_selection"] = "(pt_1 > 40) & (pt_2 > 40)"
    cuts["trg_selection"] = TRIGGER[channel]
    return cuts


def skim_cuts(channel: str) -> dict[str, str]:
    """Baseline without the charge requirement and with the loosest tau isolation, so that all estimation regions survive."""
    cuts = {name: expr for name, expr in baseline_cuts(channel).items() if name != "os"}
    cuts["tau_iso"] = " & ".join(loose(leg) for leg in legs(channel))
    return cuts


def anti_iso_cut(channel: str) -> str:
    if channel in LT_CHANNELS:
        return anti_isolated(2)
    return f"(({isolated(1)} & {anti_isolated(2)}) | ({anti_isolated(1)} & {isolated(2)}))"


def fake_factor_weight(channel: str) -> str:
    if channel in LT_CHANNELS:
        return FF_COLUMNS["lt"]
    return (
        f"0.5 * {FF_COLUMNS['tt_1']} * (id_tau_vsJet_{TAU_VS_JET_WP}_1 < 0.5)"
        f" + 0.5 * {FF_COLUMNS['tt_2']} * (id_tau_vsJet_{TAU_VS_JET_WP}_2 < 0.5)"
    )


def regions(channel: str, jet_fakes: str = "mc") -> tuple[Region, ...]:
    """Estimation regions: same-sign always; the fake-factor region (FF friend weight) in ff mode, the ABCD regions in mc mode."""
    anti = anti_iso_cut(channel)
    same_sign = Region("same_sign", replace_cuts={"os": SAME_SIGN})
    if jet_fakes == "ff":
        return (same_sign, Region("anti_iso", replace_cuts={"tau_iso": anti}, add_weights={"fake_factor": fake_factor_weight(channel)}))
    return (
        same_sign,
        Region("abcd_same_sign", replace_cuts={"os": SAME_SIGN}),
        Region("abcd_anti_iso", replace_cuts={"tau_iso": anti}),
        Region("abcd_same_sign_anti_iso", replace_cuts={"os": SAME_SIGN, "tau_iso": anti}),
    )


def channel_definition(channel: str, jet_fakes: str = "mc") -> Channel:
    return Channel(
        name=channel,
        skim=Selection(cuts=skim_cuts(channel)),
        baseline=Selection(cuts=baseline_cuts(channel), weights=baseline_weights(channel)),
        regions=regions(channel, jet_fakes),
        keep_columns=("event", "run", "lumi"),
    )


TOP_PT = {"top_pt": "topPtReweightWeight"}
_DITAU_TRIGGERS = ("double_tau35_mediumiso", "double_tau35_tightiso", "double_tau40_mediumiso", "double_tau40_tightiso")


def tau_id_weight(channel: str) -> str:
    """vsJet scale factor for genuine hadronic taus on every hadronic tau leg."""
    return " * ".join(f"((gen_match_{leg} == 5) * id_wgt_tau_vsJet_{TAU_VS_JET_WP}_{leg} + (gen_match_{leg} != 5))" for leg in legs(channel))


def tau_vs_lepton_weights(channel: str) -> dict[str, str]:
    return {
        "vs_mu": " * ".join(f"id_wgt_tau_vsMu_{TAU_VS_MU_WP[channel]}_{leg}" for leg in legs(channel)),
        "vs_ele": " * ".join(f"id_wgt_tau_vsEle_{TAU_VS_ELE_WP[channel]}_{leg}" for leg in legs(channel)),
    }


def lepton_weights(channel: str) -> dict[str, str]:
    if channel == "mt":
        return {"id": "id_wgt_mu_1", "iso": "iso_wgt_mu_1"}
    if channel == "et":
        return {"id": "id_wgt_ele_1"}  # the v15 ntuples carry no separate electron isolation SF
    return {}


def tt_trigger_weight() -> str:
    """Leg product of the first fired di-tau trigger (priority 35 mediumiso, 35 tightiso, 40 mediumiso, 40 tightiso)."""
    terms = []
    for index, trigger in enumerate(_DITAU_TRIGGERS):
        conditions = [f"(trg_{earlier} < 0.5)" for earlier in _DITAU_TRIGGERS[:index]] + [f"(trg_{trigger} > 0.5)"]
        # booleans are combined with & and multiplied into the weight as one factor (numexpr has no bool * bool)
        terms.append("(" + " & ".join(conditions) + f") * (trg_wgt{trigger}_leg1 * trg_wgt{trigger}_leg2)")
    return "(" + " + ".join(terms) + ")"


def trigger_weight(channel: str) -> dict[str, str]:
    return {"trigger": {"mt": "trg_wgt_single_mu24", "et": "trg_wgt_single_ele32"}.get(channel, tt_trigger_weight())}


def baseline_weights(channel: str) -> dict[str, str]:
    """Weights applied to every simulated process (the core applies them to MC only)."""
    return {
        "puweight": "puweight",
        **lepton_weights(channel),
        "tau_id": tau_id_weight(channel),
        **tau_vs_lepton_weights(channel),
        **trigger_weight(channel),
        "btag": "btag_weight_upart",
    }


def genmatch_cuts(channel: str) -> dict[str, str]:
    """T = genuine di-tau, J = jet -> tau_h fake, L = everything else (lepton fakes)."""
    true_tau = {
        "mt": "((gen_match_1 == 4) & (gen_match_2 == 5))",
        "et": "((gen_match_1 == 3) & (gen_match_2 == 5))",
        "tt": "((gen_match_1 == 5) & (gen_match_2 == 5))",
    }[channel]
    jet_fake = "((gen_match_1 == 6) | (gen_match_2 == 6))" if channel == "tt" else "(gen_match_2 == 6)"
    return {"T": true_tau, "L": f"(~{true_tau} & ~{jet_fake})", "J": jet_fake}


def embedding_weights(channel: str) -> dict[str, str]:
    """Weights of the embedded sample (prepared; column names unverified until embedding ntuples exist)."""
    return {
        "emb_genweight": "emb_genweight",
        "emb_selection": "emb_idsel_wgt_1 * emb_idsel_wgt_2 * emb_triggersel_wgt",
        **lepton_weights(channel),
        "tau_id": tau_id_weight(channel),
        **trigger_weight(channel),
    }
