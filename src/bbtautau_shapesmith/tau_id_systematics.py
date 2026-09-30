"""MC shape uncertainties of the tau-ID/ES measurement: the predecessor's families with MorphingTauID2017's names.

The names are the datacard names of HttSystematics_TauIDRun2.cc of the patched TauIDSFMeasurement. Each is filled from
the CROWN shift `<quantity>__<shift>` (variations package) of sm_tau_id_measurement_config or by a weight, as in smhtt_ul
config/shapes/variations.py. They are MC only, mt only (the mm control region carries normalisation uncertainties only)
and filled in the nominal region, as in the predecessor. Its dm1 bug is fixed: CMS_scale_t_dm1 uses the DM1 shifts, not
the DM0 one.

CROWN produces finer shifts than the predecessor's families: the tau ES per decay mode in three pT bins and the muon ->
tau ES in five muon-chamber wheels. Each family is one VariationSum of those shifts, exact since an event has one tau.
The jet energy scale is the regrouped set plus HEM; the patch declares it instead of jvoss's 28 individual sources,
which CROWN no longer produces. Shifts that move a column of the measurement but have no shape in MorphingTauID2017
(vsEle SF, electron -> tau ES, JER, electron ES) are filled under their CROWN name, so that the skim declares every
shifted branch; the datacards ignore them.

The vsJet SF: the predecessor's CMS_eff_t_dm{0,1,10,11} shapes equal the nominal (its CROWN shift evaluated the
nominal SF), and the payload was fitted so; they stay no-ops here, and the production has no vsJet SF shifts.
"""
from __future__ import annotations

from shapesmith.histogram import part_of
from shapesmith.model import ColumnVariation, VariationSum, WeightVariation

ERA_TAG = "Run2018"
DIRECTIONS = ("Up", "Down")
DECAY_MODES = (0, 1, 10, 11)
TAU_ES_PT_BINS = ("pt20to40", "pt40to60", "pt60toInf")
WHEELS = (1, 2, 3, 4, 5)
ETA_REGIONS = ("barrel", "endcap")

# datacard name -> CROWN jet energy scale shift without its direction: the regrouped sources, uncorrelated and correlated
# between eras, and HEM
JES_SHIFTS = {
    **{f"CMS_scale_j_{source}_{ERA_TAG}": f"CMS_scale_j_{source}_2018" for source in ("Absolute", "BBEC1", "EC2", "HF", "RelativeSample")},
    **{f"CMS_scale_j_{source}": f"CMS_scale_j_{source}" for source in ("Absolute", "BBEC1", "EC2", "HF", "FlavorQCD", "RelativeBal")},
    f"CMS_scale_j_HEMIssue_{ERA_TAG}": "CMS_HEM_2018",
}

# (datacard name, CROWN shift without its direction, sample groups or None for every MC group)
CROWN_SHIFTS = (
    *((name, shift, None) for name, shift in JES_SHIFTS.items()),
    (f"CMS_scale_met_unclustered_energy_{ERA_TAG}", "CMS_scale_met_unclustered_energy_2018", None),
    (f"CMS_res_met_{ERA_TAG}", "CMS_res_met_RecoilCalibration_2018", ("DY", "W")),
    (f"CMS_scale_met_{ERA_TAG}", "CMS_scale_met_RecoilCalibration_2018", ("DY", "W")),
    ("CMS_PileUp", "CMS_pileup_2018", None),
    (f"CMS_eff_m_trigger_{ERA_TAG}", "CMS_eff_m_trigger_2018", None),
    *((f"CMS_fake_m_WH{wheel}_{ERA_TAG}", f"CMS_fake_t_DeepTau2018v2p5_VSmu_wheel{wheel}_2018", None) for wheel in WHEELS),
)

# datacard name -> part -> CROWN shift without its direction: one family summed from its CROWN shifts
SUMMED_SHIFTS = {
    **{f"CMS_scale_t_dm{dm}_{ERA_TAG}": {pt: f"CMS_scale_t_DeepTau2018v2p5_DM{dm}_{pt}_genTau_2018" for pt in TAU_ES_PT_BINS} for dm in DECAY_MODES},
    f"CMS_scale_fake_m_{ERA_TAG}": {f"wheel{wheel}": f"CMS_scale_t_DeepTau2018v2p5_genMuon_wheel{wheel}_2018" for wheel in WHEELS},
}

# CROWN shifts without a shape in MorphingTauID2017, filled under their own name
UNUSED_SHIFTS = (
    *(f"CMS_fake_t_DeepTau2018v2p5_VSe_DM{dm}_{region}_2018" for dm in DECAY_MODES for region in ETA_REGIONS),
    *(f"CMS_scale_t_DeepTau2018v2p5_DM{dm}_genElectron_{region}_2018" for dm in DECAY_MODES for region in ETA_REGIONS),
    "CMS_res_j_2018",
    "CMS_scale_e_2018",
    "CMS_res_e_2018",
)

# (datacard name, weight name, Up expression, Down expression): the weight's carriers get the variation. The jet fake
# rate varies by max(1 - 0.002 pT, 0.6) and min(1 + 0.002 pT, 1.4), both constant from 200 GeV.
WEIGHT_SHIFTS = (
    (f"CMS_fake_j_{ERA_TAG}", "jet_fake", "(1.0 - pt_2 * 0.002) * (pt_2 < 200) + 0.6 * (pt_2 >= 200)", "(1.0 + pt_2 * 0.002) * (pt_2 < 200) + 1.4 * (pt_2 >= 200)"),
    ("CMS_htt_ttbarShape", "top_pt", "topPtReweightWeight * topPtReweightWeight", "1.0"),
)


def mc_variations(tau_id_weight: str) -> tuple[ColumnVariation | WeightVariation, ...]:
    """Every MC variation, the parts of the summed families included; `tau_id_weight` is the nominal vsJet SF weight,
    which the no-op CMS_eff_t_dm* keep."""
    shifts = (*CROWN_SHIFTS, *((shift, shift, None) for shift in UNUSED_SHIFTS))
    columns = tuple(ColumnVariation(f"{name}{d}", f"__{shift}{d}", groups=groups, regions=("nominal",)) for name, shift, groups in shifts for d in DIRECTIONS)
    parts = tuple(
        ColumnVariation(part_of(f"{name}{d}", part), f"__{shift}{d}", regions=("nominal",)) for name, family in SUMMED_SHIFTS.items() for part, shift in family.items() for d in DIRECTIONS
    )
    weights = tuple(WeightVariation(f"{name}{d}", {weight: expr}, regions=("nominal",)) for name, weight, *exprs in WEIGHT_SHIFTS for d, expr in zip(DIRECTIONS, exprs))
    no_ops = tuple(WeightVariation(f"CMS_eff_t_dm{dm}_{ERA_TAG}{d}", {"tau_id": tau_id_weight}, regions=("nominal",)) for dm in DECAY_MODES for d in DIRECTIONS)
    return columns + parts + weights + no_ops


def mc_variation_sums() -> tuple[VariationSum, ...]:
    """The summed families: the tau ES per decay mode over its pT bins, the muon -> tau ES over the wheels."""
    return tuple(VariationSum(name, tuple(family)) for name, family in SUMMED_SHIFTS.items())
