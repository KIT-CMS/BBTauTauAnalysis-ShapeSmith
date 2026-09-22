from bbtautau_shapesmith.constants import BTAG_COMPONENTS, DILEPTON_CHANNELS, TAU_CHANNELS
from tests.helpers import branches, nicks


def test_inventory_nicks_are_unique():
    for sample_list in ("sm2018_binned_v1", "sm2018_binned_v2"):
        inventory = nicks(sample_list)
        assert inventory and len(inventory) == len(set(inventory))


def test_btag_components_exist_in_every_channel():
    for channel in TAU_CHANNELS + DILEPTON_CHANNELS:
        for component in BTAG_COMPONENTS:
            assert {f"btag_weight_upart_up_{component}", f"btag_weight_upart_down_{component}"} <= branches(channel), (channel, component)
