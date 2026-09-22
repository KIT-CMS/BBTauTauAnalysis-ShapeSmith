from bbtautau_shapesmith.constants import BTAG_COMPONENTS, DILEPTON_CHANNELS, TAU_CHANNELS
from tests.helpers import branches, nicks


def test_nick_list():
    for sample_list in ("sm2018_binned_v1", "sm2018_binned_v2"):
        assert len(nicks(sample_list)) == 79 and len(set(nicks(sample_list))) == 79
    assert not set(nicks("sm2018_binned_v1")) & set(nicks("sm2018_binned_v2"))  # renamed between the productions


def test_branch_fixtures_have_v15_columns_and_no_shifts():
    assert {"n_bjets", "n_jets", "btag_weight_upart", "dilepton_veto", "trg_single_mu24", "trg_wgt_single_mu24", "genWeight", "event"} <= branches("mt")
    assert {"trg_single_ele32", "trg_wgt_single_ele32", "id_wgt_ele_1"} <= branches("et")
    assert {"trg_double_tau35_mediumiso", "trg_wgtdouble_tau35_mediumiso_leg1", "id_wgt_tau_vsJet_Medium_1"} <= branches("tt")
    assert {"id_wgt_ele_1", "reco_wgt_ele_1", "id_wgt_mu_2", "iso_wgt_mu_2", "trg_wgt_single_mu24"} <= branches("em")
    assert {"id_wgt_mu_1", "iso_wgt_mu_2", "trg_single_mu24"} <= branches("mm")
    assert {"reco_wgt_ele_1", "reco_wgt_ele_2", "trg_single_ele32"} <= branches("ee")
    for channel in TAU_CHANNELS + DILEPTON_CHANNELS:
        assert not [c for c in branches(channel) if "__" in c]


def test_btag_components_exist_in_every_channel():
    for channel in TAU_CHANNELS + DILEPTON_CHANNELS:
        for component in BTAG_COMPONENTS:
            assert {f"btag_weight_upart_up_{component}", f"btag_weight_upart_down_{component}"} <= branches(channel), (channel, component)
    assert len(BTAG_COMPONENTS) == 40
