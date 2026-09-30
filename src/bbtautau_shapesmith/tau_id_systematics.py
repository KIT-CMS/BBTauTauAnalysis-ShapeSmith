"""MC shape uncertainties of the tau-ID/ES measurement: the predecessor's set.

The names are the datacard names of MorphingTauID2017 (HttSystematics_TauIDRun2.cc of the patched
TauIDSFMeasurement); each is filled as in smhtt_ul config/shapes/variations.py, from the CROWN shift
`<quantity>__<shift>` of the measurement production or by a weight. They are MC only, mt only (the mm control region
carries normalisation uncertainties only) and filled in the nominal region, as in the predecessor. Its dm1 bug is
fixed: CMS_scale_t_dm1 and CMS_eff_t_dm1 use the DM1 shifts, not the DM0 ones.
"""
from __future__ import annotations

from shapesmith.model import ColumnVariation, WeightVariation

ERA_TAG = "Run2018"
DIRECTIONS = ("Up", "Down")

# datacard name -> CROWN jet energy scale source (shift jesUnc<source>)
JES_SOURCES = {
    "CMS_scale_j_Total": "Total", "CMS_scale_j_SinglePionECAL": "SinglePionECAL", "CMS_scale_j_SinglePionHCAL": "SinglePionHCAL",
    "CMS_scale_j_AbsoluteMPFBias": "AbsoluteMPFBias", "CMS_scale_j_AbsoluteScale": "AbsoluteScale", "CMS_scale_j_Fragmentation": "Fragmentation",
    "CMS_scale_j_PileUpDataMC": "PileUpDataMC", "CMS_scale_j_RelativeFSR": "RelativeFSR", "CMS_scale_j_PileupPtRef": "PileUpPtRef",
    "CMS_scale_j_AbsoluteStat": "AbsoluteStat", "CMS_scale_j_TimePtEta": "TimePtEta", "CMS_scale_j_RelativeStatFSR": "RelativeStatFSR",
    "CMS_scale_j_FlavorQCD": "FlavorQCD", "CMS_scale_j_PileupPtEC1": "PileUpPtEC1", "CMS_scale_j_PileUpPtBB": "PileUpPtBB",
    "CMS_scale_j_RelativePtBB": "RelativePtBB", "CMS_scale_j_RelativeJEREC1": "RelativeJEREC1", "CMS_scale_j_RelativePtEC1": "RelativePtEC1",
    "CMS_scale_j_RelativeStatEC": "RelativeStatEC", "CMS_scale_j_RelativePtHF": "RelativePtHF", "CMS_scale_j_PileUpPtHF": "PileUpPtHF",
    "CMS_scale_j_RelativeJERHF": "RelativeJERHF", "CMS_scale_j_RelativeStatHF": "RelativeStatHF", "CMS_scale_j_PileUpPtEC2": "PileUpPtEC2",
    "CMS_scale_j_RelativeJEREC2": "RelativeJEREC2", "CMS_scale_j_RelativePtEC2": "RelativePtEC2", "CMS_scale_j_RelativeBal": "RelativeBal",
    "CMS_scale_j_RelativeSample": "RelativeSample", f"CMS_scale_j_HEMIssue_{ERA_TAG}": "HEMIssue",
}

# (datacard name, CROWN shift without its direction, sample groups or None for every MC group)
CROWN_SHIFTS = (
    *((name, f"jesUnc{source}", None) for name, source in JES_SOURCES.items()),
    (f"CMS_scale_met_unclustered_energy_{ERA_TAG}", "metUnclusteredEn", None),
    (f"CMS_res_met_{ERA_TAG}", "metRecoilResol", ("DY", "W")),
    (f"CMS_scale_met_{ERA_TAG}", "metRecoilResp", ("DY", "W")),
    ("CMS_PileUp", "PileUp", None),
    (f"CMS_eff_m_trigger_{ERA_TAG}", "singleMuonTriggerSF", None),
    (f"CMS_scale_t_dm0_{ERA_TAG}", "tauEs1prong0pizero", None),
    (f"CMS_scale_t_dm1_{ERA_TAG}", "tauEs1prong1pizero", None),
    (f"CMS_scale_t_dm10_{ERA_TAG}", "tauEs3prong0pizero", None),
    (f"CMS_scale_t_dm11_{ERA_TAG}", "tauEs3prong1pizero", None),
    (f"CMS_eff_t_dm0_{ERA_TAG}", "vsJetTauDM0", None),
    (f"CMS_eff_t_dm1_{ERA_TAG}", "vsJetTauDM1", None),
    (f"CMS_eff_t_dm10_{ERA_TAG}", "vsJetTauDM10", None),
    (f"CMS_eff_t_dm11_{ERA_TAG}", "vsJetTauDM11", None),
    (f"CMS_scale_fake_m_{ERA_TAG}", "tauMuFakeEs", ("DY",)),
    *((f"CMS_fake_m_WH{wheel}_{ERA_TAG}", f"vsMuWheel{wheel}", None) for wheel in range(1, 6)),
)

# (datacard name, weight name, Up expression, Down expression): the weight's carriers get the variation. The jet fake
# rate varies by max(1 - 0.002 pT, 0.6) and min(1 + 0.002 pT, 1.4), both constant from 200 GeV.
WEIGHT_SHIFTS = (
    (f"CMS_fake_j_{ERA_TAG}", "jet_fake", "(1.0 - pt_2 * 0.002) * (pt_2 < 200) + 0.6 * (pt_2 >= 200)", "(1.0 + pt_2 * 0.002) * (pt_2 < 200) + 1.4 * (pt_2 >= 200)"),
    ("CMS_htt_ttbarShape", "top_pt", "topPtReweightWeight * topPtReweightWeight", "1.0"),
)


def mc_variations() -> tuple[ColumnVariation | WeightVariation, ...]:
    columns = tuple(ColumnVariation(f"{name}{d}", f"__{shift}{d}", groups=groups, regions=("nominal",)) for name, shift, groups in CROWN_SHIFTS for d in DIRECTIONS)
    weights = tuple(WeightVariation(f"{name}{d}", {weight: expr}, regions=("nominal",)) for name, weight, *exprs in WEIGHT_SHIFTS for d, expr in zip(DIRECTIONS, exprs))
    return columns + weights
