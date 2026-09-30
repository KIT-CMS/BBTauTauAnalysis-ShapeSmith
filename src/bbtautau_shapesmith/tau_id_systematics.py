"""MC shape uncertainties of the tau-ID/ES measurement: the predecessor's set.

The names are the datacard names of MorphingTauID2017 (HttSystematics_TauIDRun2.cc of the patched
TauIDSFMeasurement); each is filled as in smhtt_ul config/shapes/variations.py, from the CROWN shift
`<quantity>__<shift>` of sm_tau_id_measurement_config or by a weight. They are MC only, mt only (the mm control
region carries normalisation uncertainties only) and filled in the nominal region, as in the predecessor. Its dm1
bug is fixed: CMS_scale_t_dm1 uses the DM1 shift, not the DM0 one.

The vsJet SF: the predecessor's CMS_eff_t_dm{0,1,10,11} shapes equal the nominal (its CROWN shift evaluated the
nominal SF), and the payload was fitted so; they stay no-ops here. The production carries the components of the POG
Run-2 v15 SF instead; they are read as their own variations, which no datacard uses until it is decided how they
enter.
"""
from __future__ import annotations

from shapesmith.model import ColumnVariation, WeightVariation

ERA_TAG = "Run2018"
DIRECTIONS = ("Up", "Down")
DECAY_MODES = (0, 1, 10, 11)

# datacard name -> CROWN jet energy scale shift without its direction
JES_SHIFTS = {
    "CMS_scale_j_Total": "jesUncTotal", "CMS_scale_j_SinglePionECAL": "jesUncSinglePionECAL", "CMS_scale_j_SinglePionHCAL": "jesUncSinglePionHCAL",
    "CMS_scale_j_AbsoluteMPFBias": "jesUncAbsoluteMPFBias", "CMS_scale_j_AbsoluteScale": "jesUncAbsoluteScale", "CMS_scale_j_Fragmentation": "jesUncFragmentation",
    "CMS_scale_j_PileUpDataMC": "jesUncPileUpDataMC", "CMS_scale_j_RelativeFSR": "jesUncRelativeFSR", "CMS_scale_j_PileupPtRef": "jesUncPileUpPtRef",
    "CMS_scale_j_AbsoluteStat": "jesUncAbsoluteStat", "CMS_scale_j_TimePtEta": "jesUncTimePtEta", "CMS_scale_j_RelativeStatFSR": "jesUncRelativeStatFSR",
    "CMS_scale_j_FlavorQCD": "jesUncFlavorQCD", "CMS_scale_j_PileupPtEC1": "jesUncPileUpPtEC1", "CMS_scale_j_PileUpPtBB": "jesUncPileUpPtBB",
    "CMS_scale_j_RelativePtBB": "jesUncRelativePtBB", "CMS_scale_j_RelativeJEREC1": "jesUncRelativeJEREC1", "CMS_scale_j_RelativePtEC1": "jesUncRelativePtEC1",
    "CMS_scale_j_RelativeStatEC": "jesUncRelativeStatEC", "CMS_scale_j_RelativePtHF": "jesUncRelativePtHF", "CMS_scale_j_PileUpPtHF": "jesUncPileUpPtHF",
    "CMS_scale_j_RelativeJERHF": "jesUncRelativeJERHF", "CMS_scale_j_RelativeStatHF": "jesUncRelativeStatHF", "CMS_scale_j_PileUpPtEC2": "jesUncPileUpPtEC2",
    "CMS_scale_j_RelativeJEREC2": "jesUncRelativeJEREC2", "CMS_scale_j_RelativePtEC2": "jesUncRelativePtEC2", "CMS_scale_j_RelativeBal": "jesUncRelativeBal",
    "CMS_scale_j_RelativeSample": "jesUncRelativeSample2018", f"CMS_scale_j_HEMIssue_{ERA_TAG}": "jesUncHEMIssue",
}
# the POG vsJet SF components: per decay mode, and correlated across decay modes (provisional names, not in datacards)
VS_JET_COMPONENTS = {
    **{f"CMS_eff_t_{component.lower()}_dm{dm}_{ERA_TAG}": f"vsJetTau{component}DM{dm}" for dm in DECAY_MODES for component in ("Stat1", "Stat2", "SystTes")},
    f"CMS_eff_t_syst_{ERA_TAG}": "vsJetTauSyst2018",
    "CMS_eff_t_syst_allEras": "vsJetTauSystAllEras",
}

# (datacard name, CROWN shift without its direction, sample groups or None for every MC group)
CROWN_SHIFTS = (
    *((name, shift, None) for name, shift in JES_SHIFTS.items()),
    (f"CMS_scale_met_unclustered_energy_{ERA_TAG}", "metUnclusteredEn", None),
    (f"CMS_res_met_{ERA_TAG}", "metRecoilResol", ("DY", "W")),
    (f"CMS_scale_met_{ERA_TAG}", "metRecoilResp", ("DY", "W")),
    ("CMS_PileUp", "PileUp", None),
    (f"CMS_eff_m_trigger_{ERA_TAG}", "singleMuonTriggerSF", None),
    (f"CMS_scale_t_dm0_{ERA_TAG}", "tauEs1prong0pizero", None),
    (f"CMS_scale_t_dm1_{ERA_TAG}", "tauEs1prong1pizero", None),
    (f"CMS_scale_t_dm10_{ERA_TAG}", "tauEs3prong0pizero", None),
    (f"CMS_scale_t_dm11_{ERA_TAG}", "tauEs3prong1pizero", None),
    (f"CMS_scale_fake_m_{ERA_TAG}", "tauMuFakeEs", ("DY",)),
    *((f"CMS_fake_m_WH{wheel}_{ERA_TAG}", f"vsMuWheel{wheel}", None) for wheel in range(1, 6)),
    *((name, shift, None) for name, shift in VS_JET_COMPONENTS.items()),
)

# (datacard name, weight name, Up expression, Down expression): the weight's carriers get the variation. The jet fake
# rate varies by max(1 - 0.002 pT, 0.6) and min(1 + 0.002 pT, 1.4), both constant from 200 GeV.
WEIGHT_SHIFTS = (
    (f"CMS_fake_j_{ERA_TAG}", "jet_fake", "(1.0 - pt_2 * 0.002) * (pt_2 < 200) + 0.6 * (pt_2 >= 200)", "(1.0 + pt_2 * 0.002) * (pt_2 < 200) + 1.4 * (pt_2 >= 200)"),
    ("CMS_htt_ttbarShape", "top_pt", "topPtReweightWeight * topPtReweightWeight", "1.0"),
)


def mc_variations(tau_id_weight: str) -> tuple[ColumnVariation | WeightVariation, ...]:
    """Every MC variation; `tau_id_weight` is the nominal vsJet SF weight, which the no-op CMS_eff_t_dm* keep."""
    columns = tuple(ColumnVariation(f"{name}{d}", f"__{shift}{d}", groups=groups, regions=("nominal",)) for name, shift, groups in CROWN_SHIFTS for d in DIRECTIONS)
    weights = tuple(WeightVariation(f"{name}{d}", {weight: expr}, regions=("nominal",)) for name, weight, *exprs in WEIGHT_SHIFTS for d, expr in zip(DIRECTIONS, exprs))
    no_ops = tuple(WeightVariation(f"CMS_eff_t_dm{dm}_{ERA_TAG}{d}", {"tau_id": tau_id_weight}, regions=("nominal",)) for dm in DECAY_MODES for d in DIRECTIONS)
    return columns + weights + no_ops
