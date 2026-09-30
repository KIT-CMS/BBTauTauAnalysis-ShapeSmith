import json

import pytest
from shapesmith.samples import normalisation, read_sample_list

from bbtautau_shapesmith.constants import BTAG_COMPONENTS, DILEPTON_CHANNELS, TAU_CHANNELS
from tests.helpers import DATABASE, INVENTORY, branches

INVENTORIES = sorted(INVENTORY.glob("*.txt"))


@pytest.mark.parametrize("path", INVENTORIES, ids=[p.stem for p in INVENTORIES])
def test_inventory_resolves_by_nick(path):
    nicks = read_sample_list(path)
    assert nicks and set(normalisation(DATABASE, nicks)) == set(nicks)


def test_fixture_database_holds_exactly_the_inventory_nicks():
    assert set(json.loads(DATABASE.read_text())) == {nick for path in INVENTORIES for nick in read_sample_list(path)}


def test_btag_components_exist_in_every_channel():
    for channel in TAU_CHANNELS + DILEPTON_CHANNELS:
        for component in BTAG_COMPONENTS:
            assert {f"btag_weight_upart_up_{component}", f"btag_weight_upart_down_{component}"} <= branches(channel), (channel, component)
