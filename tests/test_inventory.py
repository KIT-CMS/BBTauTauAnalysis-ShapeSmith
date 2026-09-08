from bbtautau_shapesmith.constants import BTAG_COMPONENTS, CHANNELS
from tests.helpers import inventory, nicks


def test_nick_list():
    for sample_list in ("sm2018_binned_v1", "sm2018_binned_v2"):
        assert len(nicks(sample_list)) == 79 and len(set(nicks(sample_list))) == 79
    assert not set(nicks("sm2018_binned_v1")) & set(nicks("sm2018_binned_v2"))  # renamed between the productions


def test_inventories_have_v15_columns_and_no_shifts():
    assert {"n_bjets", "n_jets", "btag_weight_upart", "dilepton_veto", "trg_single_mu24", "trg_wgt_single_mu24", "genWeight", "event"} <= inventory("mt")
    assert {"trg_single_ele32", "trg_wgt_single_ele32", "id_wgt_ele_1"} <= inventory("et")
    assert {"trg_double_tau35_mediumiso", "trg_wgtdouble_tau35_mediumiso_leg1", "id_wgt_tau_vsJet_Medium_1"} <= inventory("tt")
    for channel in CHANNELS:
        assert not [c for c in inventory(channel) if "__" in c]


def test_btag_components_exist():
    for component in BTAG_COMPONENTS:
        assert {f"btag_weight_upart_up_{component}", f"btag_weight_upart_down_{component}"} <= inventory("mt")
    assert len(BTAG_COMPONENTS) == 40
