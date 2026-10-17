"""MC shape uncertainties of the tau-ID/ES measurement per era: the predecessor's families with MorphingTauID2017's names.

The names are the datacard names of HttSystematics_TauIDRun2.cc of the patched TauIDSFMeasurement, whose $ERA is the
era tag Run2016preVFP, Run2016postVFP, Run2017 or Run2018 (ERA_TAG). Each is filled from the CROWN shift
`<quantity>__<shift>` (variations package; the shift names carry the CROWN era, the prefiring shift the year) of
sm_tau_id_measurement_config or by a weight, as in smhtt_ul config/shapes/variations.py. They are MC only, mt only (the mm
control region carries normalisation uncertainties only) and filled in the nominal region, as in the predecessor. Its
dm1 bug is fixed: CMS_scale_t_dm1 uses the DM1 shifts, not the DM0 one.

CROWN produces finer shifts than the predecessor's families: the tau ES per decay mode in three pT bins and the muon ->
tau ES in five muon-chamber wheels. Each family is one VariationSum of those shifts, exact since an event has one tau.
The jet energy scale is the regrouped set; the patch declares it instead of jvoss's 28 individual sources, which CROWN no
longer produces. The HEM shift is gone: 2018 vetoes the HEM events (v7 spec E29). The prefiring weight shift (2016-2018,
E11, C11) is CMS_prefiring. Shifts that move a column of the measurement but have no shape in MorphingTauID2017 (vsEle
SF, electron -> tau ES, JER, electron ES) are filled under their own name, so that the skim declares every shifted
branch; the datacards ignore them. The electron ES and resolution are correlated between the eras (EGM, K4), so their
names have no era.

The top-pT shape (K9, TOP recommendation): with against without the reweighting, the nominal being with it; the
predecessor took the weight squared as Up.

The vsJet SF: the predecessor's CMS_eff_t_dm{0,1,10,11} shapes equal the nominal (its CROWN shift evaluated the
nominal SF), and the payload was fitted so; they stay no-ops here, and the production has no vsJet SF shifts.
"""
from __future__ import annotations

from shapesmith.histogram import part_of
from shapesmith.model import ColumnVariation, VariationSum, WeightVariation

from bbtautau_shapesmith.constants import ERAS, PREFIRING_NUISANCE, YEAR

ERA_TAG = {era: f"Run{era}" for era in ERAS}
DIRECTIONS = ("Up", "Down")
DECAY_MODES = (0, 1, 10, 11)
TAU_ES_PT_BINS = ("pt20to40", "pt40to60", "pt60toInf")
WHEELS = (1, 2, 3, 4, 5)
ETA_REGIONS = ("barrel", "endcap")
JES_UNCORRELATED = ("Absolute", "BBEC1", "EC2", "HF", "RelativeSample")
JES_CORRELATED = ("Absolute", "BBEC1", "EC2", "HF", "FlavorQCD", "RelativeBal")


def jes_shifts(era: str) -> dict[str, str]:
    """Datacard name -> CROWN jet energy scale shift without its direction: the regrouped sources, uncorrelated and
    correlated between eras."""
    tag = ERA_TAG[era]
    return {
        **{f"CMS_scale_j_{source}_{tag}": f"CMS_scale_j_{source}_{era}" for source in JES_UNCORRELATED},
        **{f"CMS_scale_j_{source}": f"CMS_scale_j_{source}" for source in JES_CORRELATED},
    }


def crown_shifts(era: str) -> tuple[tuple[str, str, tuple[str, ...] | None], ...]:
    """(datacard name, CROWN shift without its direction, sample groups or None for every MC group)."""
    tag = ERA_TAG[era]
    return (
        *((name, shift, None) for name, shift in jes_shifts(era).items()),
        (f"CMS_scale_met_unclustered_energy_{tag}", f"CMS_scale_met_unclustered_energy_{era}", None),
        (f"CMS_res_met_{tag}", f"CMS_res_met_RecoilCalibration_{era}", ("DY", "W")),
        (f"CMS_scale_met_{tag}", f"CMS_scale_met_RecoilCalibration_{era}", ("DY", "W")),
        ("CMS_PileUp", f"CMS_pileup_{era}", None),
        (f"CMS_eff_m_trigger_{tag}", f"CMS_eff_m_trigger_{era}", None),
        (PREFIRING_NUISANCE, f"CMS_ecal_prefiring_{YEAR[era]}", None),
        *((f"CMS_fake_m_WH{wheel}_{tag}", f"CMS_fake_t_DeepTau2018v2p5_VSmu_wheel{wheel}_{era}", None) for wheel in WHEELS),
    )


def summed_shifts(era: str) -> dict[str, dict[str, str]]:
    """Datacard name -> part -> CROWN shift without its direction: one family summed from its CROWN shifts."""
    tag = ERA_TAG[era]
    return {
        **{f"CMS_scale_t_dm{dm}_{tag}": {pt: f"CMS_scale_t_DeepTau2018v2p5_DM{dm}_{pt}_genTau_{era}" for pt in TAU_ES_PT_BINS} for dm in DECAY_MODES},
        f"CMS_scale_fake_m_{tag}": {f"wheel{wheel}": f"CMS_scale_t_DeepTau2018v2p5_genMuon_wheel{wheel}_{era}" for wheel in WHEELS},
    }


def unused_shifts(era: str) -> tuple[tuple[str, str], ...]:
    """(name, CROWN shift) of the CROWN shifts without a shape in MorphingTauID2017, filled under their own name (the
    electron ES and resolution without era, K4)."""
    own = (
        *(f"CMS_fake_t_DeepTau2018v2p5_VSe_DM{dm}_{region}_{era}" for dm in DECAY_MODES for region in ETA_REGIONS),
        *(f"CMS_scale_t_DeepTau2018v2p5_DM{dm}_genElectron_{region}_{era}" for dm in DECAY_MODES for region in ETA_REGIONS),
        f"CMS_res_j_{era}",
    )
    return (*((shift, shift) for shift in own), ("CMS_scale_e", f"CMS_scale_e_{era}"), ("CMS_res_e", f"CMS_res_e_{era}"))


def weight_shifts(era: str) -> tuple[tuple[str, str, str, str], ...]:
    """(datacard name, weight name, Up expression, Down expression): the weight's carriers get the variation. The jet
    fake rate varies by max(1 - 0.002 pT, 0.6) and min(1 + 0.002 pT, 1.4), both constant from 200 GeV; the top-pT shape
    is with (Up, the nominal) against without (Down) the reweighting (K9)."""
    return (
        (f"CMS_fake_j_{ERA_TAG[era]}", "jet_fake", "(1.0 - pt_2 * 0.002) * (pt_2 < 200) + 0.6 * (pt_2 >= 200)", "(1.0 + pt_2 * 0.002) * (pt_2 < 200) + 1.4 * (pt_2 >= 200)"),
        ("CMS_htt_ttbarShape", "top_pt", "topPtReweightWeight", "1.0"),
    )


def mc_variations(tau_id_weight: str, era: str) -> tuple[ColumnVariation | WeightVariation, ...]:
    """Every MC variation, the parts of the summed families included; `tau_id_weight` is the nominal vsJet SF weight,
    which the no-op CMS_eff_t_dm* keep."""
    shifts = (*crown_shifts(era), *((name, shift, None) for name, shift in unused_shifts(era)))
    columns = tuple(ColumnVariation(f"{name}{d}", f"__{shift}{d}", groups=groups, regions=("nominal",)) for name, shift, groups in shifts for d in DIRECTIONS)
    parts = tuple(
        ColumnVariation(part_of(f"{name}{d}", part), f"__{shift}{d}", regions=("nominal",)) for name, family in summed_shifts(era).items() for part, shift in family.items() for d in DIRECTIONS
    )
    weights = tuple(WeightVariation(f"{name}{d}", {weight: expr}, regions=("nominal",)) for name, weight, *exprs in weight_shifts(era) for d, expr in zip(DIRECTIONS, exprs))
    no_ops = tuple(WeightVariation(f"CMS_eff_t_dm{dm}_{ERA_TAG[era]}{d}", {"tau_id": tau_id_weight}, regions=("nominal",)) for dm in DECAY_MODES for d in DIRECTIONS)
    return columns + parts + weights + no_ops


def mc_variation_sums(era: str) -> tuple[VariationSum, ...]:
    """The summed families: the tau ES per decay mode over its pT bins, the muon -> tau ES over the wheels."""
    return tuple(VariationSum(name, tuple(family)) for name, family in summed_shifts(era).items())
