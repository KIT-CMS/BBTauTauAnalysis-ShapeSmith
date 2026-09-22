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


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_weights_columns_exist(channel):
    for expr in selection.baseline_weights(channel).values():
        assert columns_in(expr) <= branches(channel), columns_in(expr) - branches(channel)
    for expr in selection.genmatch_cuts(channel).values():
        assert columns_in(expr) <= branches(channel)
    for expr in selection.embedding_weights(channel).values():
        assert columns_in(expr) <= branches(channel) | EMBEDDING_COLUMNS
