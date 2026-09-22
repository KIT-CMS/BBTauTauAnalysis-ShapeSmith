import pytest
from shapesmith.expressions import columns_in, selection_columns

from bbtautau_shapesmith import selection
from bbtautau_shapesmith.constants import TAU_CHANNELS, EMBEDDING_COLUMNS, FF_COLUMN_SET
from tests.helpers import branches


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_baseline_and_skim_columns_exist(channel):
    channel_def = selection.channel_definition(channel)
    allowed = branches(channel) | FF_COLUMN_SET
    assert selection_columns(channel_def.baseline) <= branches(channel)
    assert selection_columns(channel_def.skim) <= branches(channel)
    for region in channel_def.regions:
        for expr in list(region.replace_cuts.values()) + list(region.add_weights.values()):
            assert columns_in(expr) <= allowed, (region.name, columns_in(expr) - allowed)
    assert set(channel_def.keep_columns) <= branches(channel)


def test_v15_selection_details():
    mt = selection.baseline_cuts("mt")
    assert mt["dilepton_veto"] == "(dilepton_veto < 0.5)" and mt["b_tagging"] == "(n_bjets >= 1) & (bpair_pt_2 > 0)"
    assert mt["trg_selection"] == "(pt_2 > 30) & (pt_1 > 25) & (trg_single_mu24 > 0.5)"
    assert mt["tau_iso"] == "(id_tau_vsJet_Medium_2 > 0.5)"
    et = selection.baseline_cuts("et")
    assert "trg_single_ele32" in et["trg_selection"] and et["against_electron"] == "(id_tau_vsEle_Tight_2 > 0.5)"
    tt = selection.baseline_cuts("tt")
    assert tt["tau_iso"] == "(id_tau_vsJet_Medium_1 > 0.5) & (id_tau_vsJet_Medium_2 > 0.5)" and tt["pt_selection"] == "(pt_1 > 40) & (pt_2 > 40)"
    assert "mt_cut" not in tt and "lepton_iso" not in tt
    skim = selection.skim_cuts("mt")
    assert "os" not in skim and skim["tau_iso"] == "(id_tau_vsJet_VVVLoose_2 > 0.5)"
    assert "&&" not in " ".join(mt.values()) and "||" not in " ".join(tt.values())


def test_regions():
    abcd = {r.name: r for r in selection.regions("mt", "mc")}
    assert set(abcd) == {"same_sign", "abcd_same_sign", "abcd_anti_iso", "abcd_same_sign_anti_iso"}
    regions = {r.name: r for r in selection.regions("mt", "ff")}
    assert set(regions) == {"same_sign", "anti_iso"}
    assert regions["same_sign"].replace_cuts == {"os": "((q_1 * q_2) > 0)"}
    assert regions["anti_iso"].replace_cuts == {"tau_iso": "((id_tau_vsJet_Medium_2 < 0.5) & (id_tau_vsJet_VVVLoose_2 > 0.5))"}
    assert regions["anti_iso"].add_weights == {"fake_factor": "fake_factor"}
    assert abcd["abcd_same_sign_anti_iso"].replace_cuts == {"os": "((q_1 * q_2) > 0)", "tau_iso": regions["anti_iso"].replace_cuts["tau_iso"]}
    assert abcd["abcd_anti_iso"].add_weights == {}
    tt = {r.name: r for r in selection.regions("tt", "ff")}
    assert "id_tau_vsJet_Medium_1 > 0.5" in tt["anti_iso"].replace_cuts["tau_iso"] and "id_tau_vsJet_VVVLoose_1 > 0.5" in tt["anti_iso"].replace_cuts["tau_iso"]
    assert tt["anti_iso"].add_weights["fake_factor"] == "0.5 * fake_factor_1 * (id_tau_vsJet_Medium_1 < 0.5) + 0.5 * fake_factor_2 * (id_tau_vsJet_Medium_2 < 0.5)"


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_weights_columns_exist(channel):
    for expr in selection.baseline_weights(channel).values():
        assert columns_in(expr) <= branches(channel), columns_in(expr) - branches(channel)
    for expr in selection.genmatch_cuts(channel).values():
        assert columns_in(expr) <= branches(channel)
    for expr in selection.embedding_weights(channel).values():
        assert columns_in(expr) <= branches(channel) | EMBEDDING_COLUMNS


def test_weight_details():
    mt = selection.baseline_weights("mt")
    assert mt["btag"] == "btag_weight_upart" and mt["trigger"] == "trg_wgt_single_mu24" and mt["iso"] == "iso_wgt_mu_1"
    assert mt["tau_id"] == "((gen_match_2 == 5) * id_wgt_tau_vsJet_Medium_2 + (gen_match_2 != 5))"
    et = selection.baseline_weights("et")
    assert et["trigger"] == "trg_wgt_single_ele32" and "iso" not in et
    tt = selection.tt_trigger_weight()
    for trigger in ("double_tau35_mediumiso", "double_tau35_tightiso", "double_tau40_mediumiso", "double_tau40_tightiso"):
        assert f"(trg_{trigger} > 0.5)) * (trg_wgt{trigger}_leg1 * trg_wgt{trigger}_leg2)" in tt
    assert tt.count("(trg_double_tau35_mediumiso < 0.5)") == 3
    gm = selection.genmatch_cuts("mt")
    assert gm["T"] == "((gen_match_1 == 4) & (gen_match_2 == 5))" and gm["J"] == "(gen_match_2 == 6)"
    assert gm["L"] == f"(~{gm['T']} & ~{gm['J']})"
    assert selection.genmatch_cuts("tt")["J"] == "((gen_match_1 == 6) | (gen_match_2 == 6))"
