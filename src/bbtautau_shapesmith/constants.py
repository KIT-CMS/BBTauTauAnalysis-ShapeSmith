"""Constants of the 2018 UL NanoAODv15 HH->bbtautau analysis."""

ERA = "2018"
LUMI_PB = 59830.0
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

# DY correction (switch dy_correction: njets_btag): DY data/MC in the mm Z window of sm2018_binned_v5, rows n_bjets
# 0, 1, >= 2 (measured with 2), columns n_jets <= 2, 3, >= 4 (the tau selections have >= 2 jets), the other MC
# subtracted with ttbar times 0.888 (btag1) and 0.965 (btag2). Derive it again with scripts/derive_dy_correction.py.
DY_NJETS_BTAG = {0: (0.869, 1.013, 1.294), 1: (1.074, 1.282, 1.732), 2: (0.916, 1.133, 1.543)}

# NN output classes in training order: index = friend column predicted_class, datacard bin id = index + 1
NN_CLASS_NAMES = ("HH2B2Tau", "DY", "ST", "TT", "VV", "jetFakes", "Other")
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

# UParT b-tag shape-correction components; columns btag_weight_upart_{up,down}_<component>
BTAG_COMPONENTS = (
    "as", "correlated", "uncorrelated", "statistic", "ttbar", "pileup", "pdf", "mur", "muf", "isrdef", "fsrdef",
    "jereta0to1p93", "jereta1p93to2p5",
    "jesAbsoluteMPFBias", "jesAbsoluteScale", "jesAbsoluteStat", "jesFlavorQCD", "jesFragmentation",
    "jesPileDataMC", "jesPilePtBB", "jesPilePtEC1", "jesPilePtEC2", "jesPilePtHF", "jesPilePtRef",
    "jesRelativeBal", "jesRelativeFSR", "jesRelativeJEREC1", "jesRelativeJEREC2", "jesRelativeJERHF",
    "jesRelativePtBB", "jesRelativePtEC1", "jesRelativePtEC2", "jesRelativePtHF", "jesRelativeSample",
    "jesRelativeStatEC", "jesRelativeStatFSR", "jesRelativeStatHF", "jesSinglePionECAL", "jesSinglePionHCAL",
    "jesTimePtEta",
)
