"""The DY correction table DY_NJETS_BTAG of an era (constants.py, v7 spec 6.10, E19) from the control histograms of
the era's dilepton run (scripts/derive_dy_correction.py).

DY data/MC per cell of n_bjets 0, 1, >= 2 (the regions btag0, btag1 and btag2, each with >= 2 jets; the row >= 2 is
measured with two b tags, btag3p has its own heavy-flavour deficit) and n_jets 2, 3, >= 4 (the bins of the control
variable n_jets): (data - the other backgrounds) / DY, the channels summed, the subtracted ttbar times a factor: a
number, or the ttbar data/MC of em in the same b-tag region ("em"). Every era and each 2016 half gets its own table
from its own run; the run must not apply the correction itself (dy_correction: none).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapesmith.histogram import INCLUSIVE, NOMINAL_VARIATION, HistKey, HistogramSet
from shapesmith.model import Analysis

from bbtautau_shapesmith.constants import ERAS

BTAG_ROWS = {0: "btag0", 1: "btag1", 2: "btag2"}  # DY_NJETS_BTAG row -> dilepton region
JET_COLUMNS = ("2", "3", ">=4")


@dataclass(frozen=True)
class Cell:
    row: int
    column: str
    data: float
    other: float
    dy: float

    @property
    def factor(self) -> float:
        return (self.data - self.other) / self.dy


def jet_yields(hset: HistogramSet, channel: str, process: str, region: str) -> np.ndarray:
    """The yields of a process in n_jets 2, 3 and >= 4."""
    key = HistKey(channel, INCLUSIVE, process, region, NOMINAL_VARIATION, "n_jets")
    if key not in hset:
        raise KeyError(f"{key.path} is missing: fill the b-tag regions first (hist --control --regions all)")
    h = hset[key]
    jets = (h.edges[:-1] + h.edges[1:]) / 2
    return np.array([h.values[jets == 2].sum(), h.values[jets == 3].sum(), h.values[jets >= 4].sum()])


def em_ttbar_scale(hset: HistogramSet, analysis: Analysis, region: str) -> float:
    """The ttbar data/MC of em in a region: (data - the other backgrounds) / TT."""
    channel = analysis.channel("em")
    data = jet_yields(hset, "em", channel.data(), region).sum()
    other = sum(jet_yields(hset, "em", process, region).sum() for process in channel.backgrounds() if process != "TT")
    return float((data - other) / jet_yields(hset, "em", "TT", region).sum())


def derive(hset: HistogramSet, analysis: Analysis, channels: list[str], ttbar_scale: str | float = "em") -> tuple[dict[int, tuple[float, ...]], dict[int, float], list[Cell]]:
    """The table (row -> factors of n_jets 2, 3, >= 4, rounded to 3 digits), the ttbar factor per row and every cell."""
    if analysis.era not in ERAS:
        raise ValueError(f"the DY table is derived per era and 2016 half, not for {analysis.era}")
    table, scales, cells = {}, {}, []
    for row, region in BTAG_ROWS.items():
        scale = em_ttbar_scale(hset, analysis, region) if ttbar_scale == "em" else float(ttbar_scale)
        scales[row] = scale
        data, other, dy = np.zeros((3, len(JET_COLUMNS)))
        for name in channels:
            channel = analysis.channel(name)
            data += jet_yields(hset, name, channel.data(), region)
            dy += jet_yields(hset, name, "DY", region)
            for process in channel.backgrounds():
                if process != "DY":
                    other += jet_yields(hset, name, process, region) * (scale if process == "TT" else 1.0)
        row_cells = [Cell(row, column, float(d), float(o), float(y)) for column, d, o, y in zip(JET_COLUMNS, data, other, dy)]
        cells += row_cells
        table[row] = tuple(round(cell.factor, 3) for cell in row_cells)
    return table, scales, cells


def table_line(era: str, table: dict[int, tuple[float, ...]]) -> str:
    """The entry of the era for constants.DY_NJETS_BTAG."""
    return f'"{era}": {table},'
