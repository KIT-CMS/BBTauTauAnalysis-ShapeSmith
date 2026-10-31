"""The DY correction table DY_NJETS_BTAG of an era (constants.py, v7 spec 6.10, E19) from the control histograms of
the era's dilepton run (scripts/derive_dy_correction.py).

DY data/MC per cell of n_bjets 0, 1, >= 2 (the regions btag0, btag1 and btag2, each with >= 2 jets; the row >= 2 is
measured with two b tags, btag3p has its own heavy-flavour deficit) and n_jets 2, 3, >= 4 (the bins of the control
variable n_jets): (data - the other backgrounds) / DY, the channels summed, the subtracted ttbar times a factor: a
number, or the ttbar data/MC of em in the same b-tag region ("em"). Every era and each 2016 half gets its own table
from its own run; the run must not apply the correction itself (dy_correction: none).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapesmith.histogram import INCLUSIVE, NOMINAL_VARIATION, HistKey, HistogramSet
from shapesmith.model import Analysis

from bbtautau_shapesmith.constants import DY_NJETS_BTAG, ERAS

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


def jet_cells(hset: HistogramSet, channel: str, process: str, region: str) -> tuple[np.ndarray, np.ndarray]:
    """The yields of a process in n_jets 2, 3 and >= 4 and their variances."""
    key = HistKey(channel, INCLUSIVE, process, region, NOMINAL_VARIATION, "n_jets")
    if key not in hset:
        raise KeyError(f"{key.path} is missing: fill the b-tag regions first (hist --control --regions all)")
    h = hset[key]
    jets = (h.edges[:-1] + h.edges[1:]) / 2
    cells = (jets == 2, jets == 3, jets >= 4)
    return np.array([h.values[c].sum() for c in cells]), np.array([h.variances[c].sum() for c in cells])


def jet_yields(hset: HistogramSet, channel: str, process: str, region: str) -> np.ndarray:
    """The yields of a process in n_jets 2, 3 and >= 4."""
    return jet_cells(hset, channel, process, region)[0]


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


# The DY-table nuisance (spec 6.7, family E): the uncertainty of the table on the DY of the categories of the
# fake-factor runs, whose nominal category of every tau channel and em_tt of em ask for a b pair with >= 1 b tag. Its
# DY lies in the rows n_bjets 1 (dilepton region btag1) and >= 2 (btag2, and btag3p, which takes the row >= 2).
NOMINAL_REGIONS = {1: ("btag1",), 2: ("btag2", "btag3p")}
# the regions of the measured rows of the nominal category, where the closure per lepton flavour is tested (btag3p has
# its own heavy-flavour deficit, which the table does not claim to describe)
CLOSURE_REGIONS = {1: "btag1", 2: "btag2"}


@dataclass(frozen=True)
class TableUncertainty:
    """The uncertainty of the DY table of an era on the DY of the nominal category, relative:
    - stat: the statistical uncertainty of the table cells (data Poisson, MC statistics of DY and of the subtracted
      backgrounds, mm + ee), propagated to the DY of the nominal category with the cell weights (the uncorrected DY of
      mm + ee in its regions);
    - residuals: per lepton flavour (mm, ee) the closure of the DY with the table, (data - other backgrounds) / (table x
      DY) - 1 summed over the cells of btag1 and btag2, with its statistical error. The table is derived on mm + ee, so
      that their sum closes by construction; the flavours test how much of the table is lepton-specific (a tau channel
      has an electron, a muon or neither).
    The nuisance is total = stat (+) the larger |residual| (in quadrature)."""

    era: str
    stat: float
    residuals: dict[str, tuple[float, float]]

    @property
    def residual(self) -> float:
        return max(abs(value) for value, _ in self.residuals.values())

    @property
    def total(self) -> float:
        return math.hypot(self.stat, self.residual)

    @property
    def lnn(self) -> float:
        return round(1 + self.total, 3)


def _scaled(hset: HistogramSet, analysis: Analysis, name: str, region: str, scale: float) -> dict[str, np.ndarray]:
    """data, other backgrounds (ttbar times `scale`) and DY of a channel and region per jet cell, with variances."""
    channel = analysis.channel(name)
    result = {key: np.zeros(len(JET_COLUMNS)) for key in ("data", "data_var", "other", "other_var", "dy", "dy_var")}
    result["data"], result["data_var"] = jet_cells(hset, name, channel.data(), region)
    for process in channel.backgrounds():
        values, variances = jet_cells(hset, name, process, region)
        if process == "DY":
            result["dy"] += values
            result["dy_var"] += variances
            continue
        factor = scale if process == "TT" else 1.0
        result["other"] += factor * values
        result["other_var"] += factor * factor * variances
    return result


def table_uncertainty(hset: HistogramSet, analysis: Analysis, channels: tuple[str, ...] = ("mm", "ee"), table: dict[int, tuple[float, ...]] | None = None) -> TableUncertainty:
    """The TableUncertainty of the era from the uncorrected control histograms of its dilepton run (the run of the
    derivation, dy_correction: none), with the ttbar scaled by the em data/MC of each b-tag region as in the derivation;
    `table` is the applied table (default: DY_NJETS_BTAG of the era)."""
    table = DY_NJETS_BTAG[analysis.era] if table is None else table
    numerator = variance = 0.0
    residual_parts = {name: np.zeros(3) for name in channels}  # per flavour: data - other, its variance, table x DY
    residual_dy_var = dict.fromkeys(channels, 0.0)
    for row, regions in NOMINAL_REGIONS.items():
        scale = em_ttbar_scale(hset, analysis, CLOSURE_REGIONS[row])
        factors = np.array(table[row])
        # the cells of the row: the statistical error of each factor from its own region (btag1, btag2)
        measured = {name: _scaled(hset, analysis, name, CLOSURE_REGIONS[row], scale) for name in channels}
        dy = sum(m["dy"] for m in measured.values())
        error = np.sqrt(sum(m["data_var"] + m["other_var"] for m in measured.values()) + factors**2 * sum(m["dy_var"] for m in measured.values())) / dy
        weights = sum(jet_yields(hset, name, "DY", region) for name in channels for region in regions)
        numerator += float(np.sum(weights * factors))
        variance += float(np.sum((weights * error) ** 2))
        for name, m in measured.items():
            residual_parts[name] += (np.sum(m["data"] - m["other"]), np.sum(m["data_var"] + m["other_var"]), np.sum(factors * m["dy"]))
            residual_dy_var[name] += float(np.sum(factors**2 * m["dy_var"]))
    residuals = {}
    for name, (excess, excess_var, predicted) in residual_parts.items():
        ratio = excess / predicted
        residuals[name] = (float(ratio - 1), float(math.sqrt(excess_var + ratio**2 * residual_dy_var[name]) / predicted))
    return TableUncertainty(analysis.era, math.sqrt(variance) / numerator, residuals)
