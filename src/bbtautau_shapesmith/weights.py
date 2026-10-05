"""Event weights: MC scale factors, the embedded sample's weights, the fake factors (v15 branch names)."""
from __future__ import annotations

from bbtautau_shapesmith.constants import FF_COLUMNS, FF_RAW_COLUMNS, LT_CHANNELS, TAU_LEGS, TAU_VS_ELE_WP, TAU_VS_JET_WP, TAU_VS_MU_WP

TOP_PT = {"top_pt": "topPtReweightWeight"}  # only ttbar ntuples carry it
_DITAU_TRIGGERS = ("double_tau35_mediumiso", "double_tau35_tightiso", "double_tau40_mediumiso", "double_tau40_tightiso")


def tau_id_weight(channel: str) -> str:
    """vsJet scale factor for genuine hadronic taus on every hadronic tau leg."""
    return " * ".join(f"((gen_match_{leg} == 5) * id_wgt_tau_vsJet_{TAU_VS_JET_WP}_{leg} + (gen_match_{leg} != 5))" for leg in TAU_LEGS[channel])


def fail_tau_weights(channel: str) -> dict[str, str]:
    # A passing-WP SF is not an inefficiency SF. Apply it only to legs which pass;
    # failed genuine taus remain uncorrected until a dedicated efficiency treatment exists.
    return {"tau_id": " * ".join(
        f"(((gen_match_{leg} == 5) & (id_tau_vsJet_{TAU_VS_JET_WP}_{leg} > 0.5)) * id_wgt_tau_vsJet_{TAU_VS_JET_WP}_{leg}"
        f" + ((gen_match_{leg} != 5) | (id_tau_vsJet_{TAU_VS_JET_WP}_{leg} < 0.5)))"
        for leg in TAU_LEGS[channel])}


def tau_vs_lepton_weights(channel: str) -> dict[str, str]:
    return {
        "vs_mu": " * ".join(f"id_wgt_tau_vsMu_{TAU_VS_MU_WP[channel]}_{leg}" for leg in TAU_LEGS[channel]),
        "vs_ele": " * ".join(f"id_wgt_tau_vsEle_{TAU_VS_ELE_WP[channel]}_{leg}" for leg in TAU_LEGS[channel]),
    }


def lepton_weights(channel: str) -> dict[str, str]:
    if channel == "mt":
        return {"id": "id_wgt_mu_1", "iso": "iso_wgt_mu_1"}
    if channel == "et":
        return {"id": "id_wgt_ele_1"}  # the MC wp90iso ID SF covers the isolation
    return {}


def tt_trigger_weight() -> str:
    """Leg product of the first fired di-tau trigger (priority 35 mediumiso, 35 tightiso, 40 mediumiso, 40 tightiso)."""
    terms = []
    for index, trigger in enumerate(_DITAU_TRIGGERS):
        conditions = [f"(trg_{earlier} < 0.5)" for earlier in _DITAU_TRIGGERS[:index]] + [f"(trg_{trigger} > 0.5)"]
        # booleans are combined with & and multiplied into the weight as one factor (numexpr has no bool * bool)
        terms.append("(" + " & ".join(conditions) + f") * (trg_wgt{trigger}_leg1 * trg_wgt{trigger}_leg2)")
    return "(" + " + ".join(terms) + ")"


def trigger_weight(channel: str) -> str:
    return {"mt": "trg_wgt_single_mu24", "et": "trg_wgt_single_ele32"}.get(channel) or tt_trigger_weight()


def mc_weights(channel: str) -> dict[str, str]:
    """Weights of every simulated process of a tau channel; process-specific weights follow them."""
    return {
        "puweight": "puweight",
        **lepton_weights(channel),
        "tau_id": tau_id_weight(channel),
        **tau_vs_lepton_weights(channel),
        "trigger": trigger_weight(channel),
        "btag": "btag_weight_upart",
    }


def embedding_weights(channel: str) -> dict[str, str]:
    """Weights of the embedded sample (Part-A output contract): no pileup or b-tag weight, the embedding ID/iso SFs of the
    light lepton (et carries an electron iso SF in embedding only). The names differ from the MC ones, so that regions
    replacing MC weights leave the embedded sample unchanged."""
    lepton = {"mt": {"id": "id_wgt_mu_1", "iso": "iso_wgt_mu_1"}, "et": {"id": "id_wgt_ele_1", "iso": "iso_wgt_ele_1"}}.get(channel, {})
    scale_factors = {**lepton, "tau_id": tau_id_weight(channel), **tau_vs_lepton_weights(channel), "trigger": trigger_weight(channel)}
    return {
        "emb_genweight": "emb_genweight",
        "emb_selection": "emb_idsel_wgt_1 * emb_idsel_wgt_2 * emb_triggersel_wgt",
        **{f"emb_{name}": expr for name, expr in scale_factors.items()},
    }


def fake_factor_weight(channel: str, ff_type: str = "corrected") -> str:
    """The FF friend weight (corrected or raw) of the anti-isolated region; in tt each failing leg carries half of its fake factor."""
    columns = FF_RAW_COLUMNS if ff_type == "raw" else FF_COLUMNS
    if channel in LT_CHANNELS:
        return columns["lt"]
    return (
        f"0.5 * {columns['tt_1']} * (id_tau_vsJet_{TAU_VS_JET_WP}_1 < 0.5)"
        f" + 0.5 * {columns['tt_2']} * (id_tau_vsJet_{TAU_VS_JET_WP}_2 < 0.5)"
    )


def dilepton_mc_weights(channel: str) -> dict[str, str]:
    """Weights of every simulated process of a light-dilepton channel (em: electron leg 1, muon leg 2)."""
    leptons = {
        "em": {"electron_id": "id_wgt_ele_1", "electron_reco": "reco_wgt_ele_1",
               "muon_id": "id_wgt_mu_2", "muon_iso": "iso_wgt_mu_2"},
        "mm": {"muon_id": "id_wgt_mu_1 * id_wgt_mu_2", "muon_iso": "iso_wgt_mu_1 * iso_wgt_mu_2"},
        "ee": {"electron_id": "id_wgt_ele_1 * id_wgt_ele_2", "electron_reco": "reco_wgt_ele_1 * reco_wgt_ele_2"},
    }[channel]
    return {"puweight": "puweight", **leptons, "btag": "btag_weight_upart",
            "trigger": "trg_wgt_single_ele32" if channel == "ee" else "trg_wgt_single_mu24"}
