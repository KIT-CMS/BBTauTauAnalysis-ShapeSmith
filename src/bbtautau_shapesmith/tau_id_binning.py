"""m_vis binning of the tau-ID/ES measurement.

The mt categories have ten bins with equal data counts in 30-160 GeV, as the predecessor's smhtt_ul
gof/build_binning.py: ShapeSmith computes them from the data of the working-point combination in each category
whenever it fills (shapesmith.binning) and records them in binning.json next to shapes.root. The mm control region is
one bin, 70-110 GeV.
"""
from shapesmith.model import EqualData

M_VIS_BINNING = EqualData(n_bins=10, low=30.0, high=160.0)
CONTROL_REGION_EDGES = (70.0, 110.0)
