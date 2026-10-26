"""Constants of the Run-2 UL NanoAODv15 HH->bbtautau analysis (2016preVFP, 2016postVFP, 2017, 2018)."""
from shapesmith.model import AnalysisError

ERAS = ("2016preVFP", "2016postVFP", "2017", "2018")
# integrated luminosity per era (v7 spec E20): 19.52 / 16.81 / 42.07 / 59.56 fb-1, the HIPM split of 2016 kept, 2017 and
# 2018 with the new normtag; Run 2 = 137.96 fb-1, labelled 138 fb-1
LUMI_PB = {"2016preVFP": 19520.0, "2016postVFP": 16810.0, "2017": 42070.0, "2018": 59560.0}
YEAR = {"2016preVFP": "2016", "2016postVFP": "2016", "2017": "2017", "2018": "2018"}  # the calendar year of an era
# an event of the n-tuples (Analysis.event_key, decided 2026-10-26): every data and embedding sample is read without
# the skim rows that repeat the event of an earlier row of the sample (2018D mutau embedding: the input files nano_35 and
# nano_53 share 44 497 events); the skims keep every row
EVENT_KEY = ("run", "lumi", "event")
# eras measured together (v7 spec E18, 6.8): one fake-factor measurement and payload of 2016 on both halves, every
# sample normalised to the luminosity of its own half
COMBINED_ERAS = {"2016": ("2016preVFP", "2016postVFP")}
TAU_CHANNELS = ("et", "mt", "tt")
LT_CHANNELS = ("et", "mt")  # tau channels with a light lepton on leg 1
TAU_LEGS = {"et": (2,), "mt": (2,), "tt": (1, 2)}  # the hadronic tau legs
DILEPTON_CHANNELS = ("em", "mm", "ee")  # light-dilepton controls: Z+jets in ee/mm, top in em
CONTROL_CHANNELS = ("em",)  # control channels of the tau analysis, in every fit: em constrains ttbar

TAU_VS_JET_WP = "Medium"
TAU_VS_JET_LOOSE_WP = "VVVLoose"  # lower edge of the anti-isolated region
TAU_VS_MU_WP = {"mt": "Tight", "et": "VLoose", "tt": "VLoose"}
TAU_VS_ELE_WP = {"mt": "VVLoose", "et": "Tight", "tt": "VVLoose"}

# b-tag multiplicity bins of the control regions (each combined with >= 2 jets)
BTAG_BINS = {"btag0": "n_bjets == 0", "btag1": "n_bjets == 1", "btag2": "n_bjets == 2", "btag3p": "n_bjets >= 3"}
# jet multiplicity bins of the dilepton diagnostic regions (switch diagnostics), each with 0 and >= 1 b tags
NJETS_BINS = {"nj2": "n_jets == 2", "nj3": "n_jets == 3", "nj4p": "n_jets >= 4"}

# DY correction (switch dy_correction: njets_btag) per era and 2016 half (E19, spec 6.10): DY data/MC in the mm and ee Z
# window, rows n_bjets 0, 1, >= 2 (measured with 2), columns n_jets <= 2, 3, >= 4 (the tau selections have >= 2 jets),
# the other MC subtracted with ttbar times the em data/MC of the same b-tag region. Derived per era with
# scripts/derive_dy_correction.py --ttbar-scale em (dy_correction.py), which prints the entry of the era.
# Every era from its smrun2_binned_v7b dilepton run (configs/smrun2_binned_v7b_dilepton.yaml, 2026-10-19: the MET fix and
# the POG muon trigger SF on MC); ttbar scale of em per row: 2016preVFP 0.832, 0.892, 0.954; 2016postVFP 0.896, 0.930,
# 1.026; 2017 0.892, 0.909, 0.981; 2018 0.855, 0.900, 0.997. They serve the v7 run YAMLs as well, so the v7 runs with
# dy_correction: njets_btag no longer reproduce exactly. The v7 tables (smrun2_binned_v7, KIT muon trigger SF on MC) are
# those of analysis abd7657: up to 3.0 % (2016preVFP), 1.6 % (2016postVFP), 0.6 % (2017) and 0.4 % (2018) higher, the
# normalisation of the mm MC, with the same n_jets shape.
DY_NJETS_BTAG = {
    "2016preVFP": {0: (0.875, 1.033, 1.355), 1: (1.052, 1.297, 1.824), 2: (0.968, 1.180, 1.631)},
    "2016postVFP": {0: (0.919, 1.094, 1.463), 1: (1.099, 1.323, 1.878), 2: (1.008, 1.230, 1.812)},
    "2017": {0: (0.884, 1.039, 1.344), 1: (1.067, 1.304, 1.726), 2: (0.938, 1.135, 1.628)},
    "2018": {0: (0.882, 1.030, 1.328), 1: (1.115, 1.334, 1.805), 2: (0.981, 1.200, 1.717)},
}
# The DY-table nuisance CMS_htt_dy_table_<era> (spec 6.7, family E): lnN on every DY process the table corrects (ZL, ZJ,
# ZTT from MC, em DY), per era and 2016 half, sized with scripts/size_dy_table_uncertainty.py on the v7b dilepton runs
# (dy_correction.table_uncertainty) for the categories of the fake-factor runs (a b pair with >= 1 b tag: rows 1 and
# >= 2, all with >= 2 jets): the table statistics propagated with the DY of btag1, btag2 and btag3p (0.45 / 0.47 / 0.29 /
# 0.26 %) in quadrature with the larger closure residual of one lepton flavour in btag1 + btag2 (mm +0.45 / +0.11 /
# +1.15 / +0.21 %, ee -0.67 / -0.13 / -1.65 / -0.26 %; 2016preVFP / 2016postVFP / 2017 / 2018).
DY_TABLE_LNN = {"2016preVFP": 1.008, "2016postVFP": 1.005, "2017": 1.017, "2018": 1.004}

# MET fix of 2026-10-18 (met_recoil_study): DY and W carry the recoil correction of the MET and its shifts
# CMS_{scale,res}_met_RecoilCalibration_<era> only in these eras (CROWN bbtautau RUN2_V15_RECOIL_CORRECTION_ERAS), from the
# productions smrun2_binned_v7b and smrun2_tau_id_v2 on; in 2016 and 2017 they take the MET of every other simulated sample
# (JES and JER propagated, no recoil shifts). smrun2_binned_v7 and smrun2_tau_id_v1 have them in every era.
RECOIL_ERAS = ("2018",)

# event selections and weights of the POG cross-check (v7 spec 6.2, 6.4)
HEM_ERAS = ("2018",)  # E29: veto of the events with a jet in the HEM 15/16 region, column hem_veto (CROWN, 2018 only)
HEM_VETO = "(hem_veto < 0.5)"
LEPTON_ETA = "(abs(eta_1) < 2.1)"  # E31: the light lepton inside the embedding acceptance (generator filter |eta| < 2.2)
PREFIRING_WEIGHT = "prefiring_wgt"  # E11: L1PreFiringWeight_Nom on MC, 2016-2018 (2018: the muon part)
PREFIRING_VARIATIONS = {"Up": "L1PreFiringWeight_Up", "Down": "L1PreFiringWeight_Dn"}  # columns of the nominal tree (C11)
PREFIRING_NUISANCE = "CMS_prefiring"  # one lnN for 2016-2018 (E11, E34)

# NN friend columns: the index of the predicted class (the class order of the ML export, ml.ML_CLASSES; datacard bin id
# = index + 1) and its score
NN_CLASS_COLUMN = "predicted_class"
NN_SCORE_COLUMN = "predicted_max_value"
NN_COLUMNS = frozenset((NN_CLASS_COLUMN, NN_SCORE_COLUMN))

# fake-factor friend (CROWN fake_factors_friend_config.py): the weight column per leg, corrected and raw (the fractions
# and the QCD and ttbar fake factors, without the DR->SR and non-closure corrections)
FF_COLUMNS = {"lt": "fake_factor", "tt_1": "fake_factor_1", "tt_2": "fake_factor_2"}
FF_RAW_COLUMNS = {"lt": "fake_factor_raw", "tt_1": "fake_factor_1_raw", "tt_2": "fake_factor_2_raw"}

# CROWN shift names of the fake-factor friend without their direction (Up/Down): every Up/Down key of the SM 2018
# payload (CROWN bbtautau 1570b43, payloads/fake_factors/sm/2018) except the SystBand{High,Low} and per-variable
# non-closure keys. The tt subleading keys carry "_subleading" after the process or correction name.
FF_SHIFTS_LT = (
    "QCDStatShift", "QCDSystBandAsym", "QCDSystMCShift",
    "QCD_DR_SR_CorrStatShift", "QCD_DR_SR_CorrSystBandAsym", "QCD_DR_SR_CorrSystMCShift",
    "QCD_non_closure_CorrStatShift", "QCD_non_closure_CorrSystBandAsym", "QCD_non_closure_CorrSystMCShift",
    "process_fractionsfrac_QCD", "process_fractionsfrac_ttbar_J",
    "ttbarStatShift", "ttbarSystBandAsym", "ttbarSystMCShift",
    "ttbar_non_closure_CorrStatShift", "ttbar_non_closure_CorrSystBandAsym", "ttbar_non_closure_CorrSystMCShift",
)
# the W+jets keys of a payload with W+jets fake factors (switch ff_processes, et and mt; v7 spec 6.8, C15)
FF_SHIFTS_W = (
    "WjetsStatShift", "WjetsSystBandAsym", "WjetsSystMCShift",
    "Wjets_DR_SR_CorrStatShift", "Wjets_DR_SR_CorrSystBandAsym", "Wjets_DR_SR_CorrSystMCShift",
    "Wjets_non_closure_CorrStatShift", "Wjets_non_closure_CorrSystBandAsym", "Wjets_non_closure_CorrSystMCShift",
    "process_fractionsfrac_Wjets",
)
FF_SHIFTS_TT = FF_SHIFTS_LT + (
    "QCD_subleadingStatShift", "QCD_subleadingSystBandAsym", "QCD_subleadingSystMCShift",
    "QCD_subleading_DR_SR_CorrStatShift", "QCD_subleading_DR_SR_CorrSystBandAsym", "QCD_subleading_DR_SR_CorrSystMCShift",
    "QCD_subleading_non_closure_CorrStatShift", "QCD_subleading_non_closure_CorrSystBandAsym", "QCD_subleading_non_closure_CorrSystMCShift",
    "process_fractions_subleadingfrac_QCD", "process_fractions_subleadingfrac_ttbar_J",
    "ttbar_subleadingStatShift", "ttbar_subleadingSystBandAsym", "ttbar_subleadingSystMCShift",
    "ttbar_subleading_non_closure_CorrStatShift", "ttbar_subleading_non_closure_CorrSystBandAsym", "ttbar_subleading_non_closure_CorrSystMCShift",
)

# Embedding tau corrections (energy scale and vsJet SF, both binned in the decay mode), one nuisance per measured
# decay-mode category: nuisance token -> the decay modes of its CROWN shifts. DM10 and DM11 are one fit, which CROWN
# shifts per decay mode; the nuisance sums the two.
EMBEDDING_TAU_DECAY_MODES = {"0": (0,), "1": (1,), "1011": (10, 11)}

# UParT fixed-WP b-tag weight variations (WPs [T, M], v7 spec C14/E14): b/c and light jets, each correlated between
# the eras and uncorrelated; columns btag_weight_upart_{up,down}_<component> (one varied, the other at its nominal)
BTAG_COMPONENTS = ("bc_correlated", "bc_uncorrelated", "light_correlated", "light_uncorrelated")


def check_era(era: str) -> str:
    """The era of the run, one of ERAS; any other is rejected."""
    if era not in ERAS:
        raise AnalysisError(f"unknown era {era!r}; the analysis supports {', '.join(ERAS)}")
    return era


def check_measurement_era(era: str) -> str:
    """The era of a measurement that can span several eras: one of ERAS or of COMBINED_ERAS."""
    if era not in ERAS and era not in COMBINED_ERAS:
        combined = ", ".join(f"{name} ({' + '.join(members)})" for name, members in COMBINED_ERAS.items())
        raise AnalysisError(f"unknown era {era!r}; the measurement supports {', '.join(ERAS)} and {combined}")
    return era


def member_eras(era: str) -> tuple[str, ...]:
    """The eras an era consists of: itself, or the members of a combined era."""
    return COMBINED_ERAS.get(era, (era,))


def lumi_pb(era: str) -> float:
    """The luminosity of an era or of a combined era (the sum of its members)."""
    return sum(LUMI_PB[member] for member in member_eras(era))


def year(era: str) -> str:
    """The calendar year of an era or of a combined era (one year)."""
    years = {YEAR[member] for member in member_eras(era)}
    if len(years) != 1:
        raise AnalysisError(f"era {era} spans the years {sorted(years)}")
    return years.pop()
