"""Event weights per era: MC scale factors, the embedded sample's weights, the fake factors (v15 branch names).

The trigger weights of the era come from triggers.TRIGGER; every MC sample carries the prefiring weight (E11: 2016,
2017 and 2018, there the muon part).
"""
from __future__ import annotations

from shapesmith.model import AnalysisError

from bbtautau_shapesmith.constants import COMBINED_ERAS, DY_NJETS_BTAG, FF_COLUMNS, FF_RAW_COLUMNS, LT_CHANNELS, LUMI_PB, PREFIRING_WEIGHT, TAU_LEGS, TAU_VS_ELE_WP, TAU_VS_JET_WP, TAU_VS_MU_WP, member_eras
from bbtautau_shapesmith.triggers import TRIGGER

TOP_PT = {"top_pt": "topPtReweightWeight"}  # only ttbar ntuples carry it
PREFIRING = {"prefiring": PREFIRING_WEIGHT}
_DY_BTAG_ROWS = {0: "(n_bjets == 0)", 1: "(n_bjets == 1)", 2: "(n_bjets >= 2)"}
_DY_JET_COLUMNS = ("(n_jets <= 2)", "(n_jets == 3)", "(n_jets >= 4)")


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
        return {"id": "id_wgt_ele_1", "reco": "reco_wgt_ele_1"}  # the MC wp90iso ID SF covers the isolation
    return {}


def trigger_weights(channel: str, era: str) -> dict[str, str]:
    """The trigger weight of the simulated samples and, for the 2017 Ele35 events, the HLT Zvtx factor (E6)."""
    trigger = TRIGGER[era][channel]
    return {"trigger": trigger.mc_weight, **({"zvtx": trigger.zvtx} if trigger.zvtx else {})}


def embedding_trigger_weights(channel: str, era: str) -> dict[str, str]:
    """The trigger weights of the embedded samples: the KIT SF of the single-lepton trigger (with the Zvtx factor of
    2017 Ele35), in tt the data efficiencies of both taus (E33)."""
    trigger = TRIGGER[era][channel]
    return {"trigger": trigger.embedding_weight, **({"zvtx": trigger.zvtx} if trigger.zvtx else {})}


def dy_table(era: str) -> dict[int, tuple[float, ...]]:
    """The DY table of an era; for a combined era (the 2016 fake-factor measurement) the luminosity-weighted mean of
    the tables of its members, which the weight of one process over the samples of both halves cannot tell apart."""
    if era in DY_NJETS_BTAG:
        return DY_NJETS_BTAG[era]
    members = member_eras(era)
    if era in COMBINED_ERAS and all(member in DY_NJETS_BTAG for member in members):
        total = sum(LUMI_PB[member] for member in members)
        return {row: tuple(round(sum(LUMI_PB[member] * DY_NJETS_BTAG[member][row][column] for member in members) / total, 4) for column in range(len(_DY_JET_COLUMNS)))
                for row in _DY_BTAG_ROWS}
    missing = ", ".join(member for member in members if member not in DY_NJETS_BTAG)
    raise AnalysisError(f"dy_correction njets_btag: no DY table for {missing} (constants.DY_NJETS_BTAG has {', '.join(DY_NJETS_BTAG)})")


def dy_njets_btag_weight(era: str) -> str:
    """The DY correction DY_NJETS_BTAG of the era (dy_table) for the event's n_bjets and n_jets, one term per cell."""
    terms = [f"({_DY_BTAG_ROWS[row]} & {jets}) * {factor}" for row, factors in dy_table(era).items() for jets, factor in zip(_DY_JET_COLUMNS, factors)]
    return "(" + " + ".join(terms) + ")"


def group_weights(group: str, era: str, dy_correction: str = "none") -> dict[str, str]:
    """Weights of one group only: the top-pT reweighting of ttbar, the DY correction (switch dy_correction)."""
    if group == "TT":
        return TOP_PT
    if group == "DY" and dy_correction == "njets_btag":
        return {"dy_correction": dy_njets_btag_weight(era)}
    return {}


def mc_weights(channel: str, era: str) -> dict[str, str]:
    """Weights of every simulated process of a tau channel; process-specific weights follow them."""
    return {
        "puweight": "puweight",
        **lepton_weights(channel),
        "tau_id": tau_id_weight(channel),
        **tau_vs_lepton_weights(channel),
        **trigger_weights(channel, era),
        **PREFIRING,
        "btag": "btag_weight_upart",
    }


def embedding_weights(channel: str, era: str) -> dict[str, str]:
    """Weights of the embedded sample (Part-A output contract): no pileup, prefiring or b-tag weight, the embedding
    ID/iso SFs of the light lepton (et carries an electron iso SF in embedding only). The names differ from the MC
    ones, so that regions replacing MC weights leave the embedded sample unchanged."""
    lepton = {"mt": {"id": "id_wgt_mu_1", "iso": "iso_wgt_mu_1"}, "et": {"id": "id_wgt_ele_1", "iso": "iso_wgt_ele_1"}}.get(channel, {})
    scale_factors = {**lepton, "tau_id": tau_id_weight(channel), **tau_vs_lepton_weights(channel), **embedding_trigger_weights(channel, era)}
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


def dilepton_mc_weights(channel: str, era: str) -> dict[str, str]:
    """Weights of every simulated process of a light-dilepton channel (em: electron leg 1, muon leg 2)."""
    leptons = {
        "em": {"electron_id": "id_wgt_ele_1", "electron_reco": "reco_wgt_ele_1",
               "muon_id": "id_wgt_mu_2", "muon_iso": "iso_wgt_mu_2"},
        "mm": {"muon_id": "id_wgt_mu_1 * id_wgt_mu_2", "muon_iso": "iso_wgt_mu_1 * iso_wgt_mu_2"},
        "ee": {"electron_id": "id_wgt_ele_1 * id_wgt_ele_2", "electron_reco": "reco_wgt_ele_1 * reco_wgt_ele_2"},
    }[channel]
    return {"puweight": "puweight", **leptons, "btag": "btag_weight_upart", **trigger_weights(channel, era), **PREFIRING}
