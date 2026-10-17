"""The DY table of an era (spec 6.10): derived from the control histograms of the era's dilepton run."""
import dataclasses

import numpy as np
import pytest
from shapesmith.histogram import INCLUSIVE, NOMINAL_VARIATION, HistKey, Histogram, HistogramSet

from bbtautau_shapesmith import dilepton, samples as samples_module
from bbtautau_shapesmith.constants import DILEPTON_CHANNELS
from bbtautau_shapesmith.dy_correction import BTAG_ROWS, derive, table_line
from tests.helpers import INVENTORY, config
from tests.run2 import list_name, write

EDGES = (1.5, 2.5, 3.5, 4.5, 10.5)  # n_jets 2, 3, 4, 5-10: the columns 2, 3 and >= 4


@pytest.fixture(scope="module")
def analysis_2017(tmp_path_factory):
    directory = tmp_path_factory.mktemp("run2_dy")
    for inventory in INVENTORY.glob("*.txt"):
        (directory / inventory.name).write_bytes(inventory.read_bytes())
    database = write(directory)
    patch = pytest.MonkeyPatch()
    patch.setattr(samples_module, "INVENTORY_DIR", directory)
    yield dilepton.build(config(DILEPTON_CHANNELS, "bbtautau_shapesmith.dilepton:build", era="2017", database=database, sample_lists=[list_name("binned", "2017")]))
    patch.undo()


def _fill(hset, channel, process, region, values):
    hset[HistKey(channel, INCLUSIVE, process, region, NOMINAL_VARIATION, "n_jets")] = Histogram(EDGES, values, values)


def histograms(analysis, dy_factor=(1.1, 1.3, 1.5), ttbar_factor=1.2):
    """DY 100 per jet bin, TT 10, every other background 1; data = the DY times dy_factor per row (mm, ee) or TT times
    ttbar_factor (em), plus the rest."""
    hset = HistogramSet()
    for name in DILEPTON_CHANNELS:
        channel = analysis.channel(name)
        for row, region in BTAG_ROWS.items():
            yields = {process: np.full(4, 100.0 if process == "DY" else 10.0 if process == "TT" else 1.0) for process in channel.backgrounds()}
            for process, values in yields.items():
                _fill(hset, name, process, region, values)
            data = sum(yields.values())
            data = data + (yields["TT"] * (ttbar_factor - 1) if name == "em" else yields["DY"] * (dy_factor[row] - 1) + yields["TT"] * (ttbar_factor - 1))
            _fill(hset, name, channel.data(), region, data)
    return hset


def test_the_table_of_an_era(analysis_2017):
    table, scales, cells = derive(histograms(analysis_2017), analysis_2017, ["mm", "ee"], "em")
    assert scales == pytest.approx({0: 1.2, 1: 1.2, 2: 1.2})  # the em ttbar data/MC of each b-tag region
    assert table == {0: (1.1, 1.1, 1.1), 1: (1.3, 1.3, 1.3), 2: (1.5, 1.5, 1.5)}
    assert [cell.column for cell in cells[:3]] == ["2", "3", ">=4"] and cells[2].dy == pytest.approx(2 * 2 * 100.0)  # >= 4: two jet bins, mm + ee
    assert table_line("2017", table) == '"2017": {0: (1.1, 1.1, 1.1), 1: (1.3, 1.3, 1.3), 2: (1.5, 1.5, 1.5)},'
    unscaled, _, _ = derive(histograms(analysis_2017), analysis_2017, ["mm", "ee"], 1.0)
    assert unscaled[0][0] == pytest.approx(1.1 + 2 * 0.2 * 10 / (2 * 100), abs=1e-3)  # the ttbar excess of mm and ee counted as DY


def test_the_table_is_per_era(analysis_2017):
    with pytest.raises(ValueError, match="per era and 2016 half, not for 2016"):
        derive(histograms(analysis_2017), dataclasses.replace(analysis_2017, era="2016"), ["mm"], 1.0)
    with pytest.raises(KeyError, match="fill the b-tag regions first"):
        derive(HistogramSet(), analysis_2017, ["mm"], 1.0)
