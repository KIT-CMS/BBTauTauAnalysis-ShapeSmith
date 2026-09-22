"""Light-dilepton control channels em/mm/ee: leg order and triggers follow CROWN's 2018 SM profile
(em: electron leg 1, muon leg 2; mm/ee: leading pT first)."""
from shapesmith.model import Channel, Region, Selection

from bbtautau_shapesmith.constants import BTAG_BINS, DILEPTON_CHANNELS


def baseline_weights(channel: str) -> dict[str, str]:
    leptons = {
        "em": {"electron_id": "id_wgt_ele_1", "electron_reco": "reco_wgt_ele_1",
               "muon_id": "id_wgt_mu_2", "muon_iso": "iso_wgt_mu_2"},
        "mm": {"muon_id": "id_wgt_mu_1 * id_wgt_mu_2", "muon_iso": "iso_wgt_mu_1 * iso_wgt_mu_2"},
        "ee": {"electron_id": "id_wgt_ele_1 * id_wgt_ele_2", "electron_reco": "reco_wgt_ele_1 * reco_wgt_ele_2"},
    }[channel]
    return {"puweight": "puweight", **leptons, "btag": "btag_weight_upart",
            "trigger": "trg_wgt_single_ele32" if channel == "ee" else "trg_wgt_single_mu24"}


def channel_definition(channel: str) -> Channel:
    if channel not in DILEPTON_CHANNELS:
        raise ValueError(f"light-dilepton channel must be one of {DILEPTON_CHANNELS}, got {channel!r}")
    cuts = {
        "os": "(q_1 * q_2) < 0",
        "lepton_iso": "(iso_1 < 0.15) & (iso_2 < 0.15)",
        "extraelec_veto": "extraelec_veto < 0.5", "extramuon_veto": "extramuon_veto < 0.5",
        # Match the effective offline thresholds inside CROWN's trigger flags.
        "pt_selection": {"em": "(pt_1 > 15) & (pt_2 > 26)",
                         "mm": "(pt_1 > 26) & (pt_2 > 15)",
                         "ee": "(pt_1 > 34) & (pt_2 > 15)"}[channel],
        "trigger": "trg_single_ele32 > 0.5" if channel == "ee" else "trg_single_mu24 > 0.5",
        "mass_window": "True" if channel == "em" else "(m_vis > 70) & (m_vis < 110)",
        "jets": "n_jets >= 2" if channel == "em" else "True",
        "b_tagging": "True",
    }
    regions = tuple(Region(name, replace_cuts={"jets": "n_jets >= 2", "b_tagging": cut})
                    for name, cut in BTAG_BINS.items())
    regions += (Region("same_sign", replace_cuts={"os": "(q_1 * q_2) > 0"}),)
    # Keep both charges so the same-sign validation is available without re-skimming.
    skim = {name: cut for name, cut in cuts.items() if name != "os"}
    return Channel(channel, Selection(cuts=skim), Selection(cuts=cuts, weights=baseline_weights(channel)),
                   regions, keep_columns=("event", "run", "lumi"))
