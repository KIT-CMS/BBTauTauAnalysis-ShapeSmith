"""Constants of the 2018 UL NanoAODv15 HH->bbtautau analysis."""

ERA = "2018"
LUMI_PB = 59830.0
TAU_CHANNELS = ("et", "mt", "tt")
LT_CHANNELS = ("et", "mt")  # tau channels with a light lepton on leg 1
DILEPTON_CHANNELS = ("em", "mm", "ee")  # light-dilepton controls: Z+jets in ee/mm, top in em

TAU_VS_JET_WP = "Medium"
TAU_VS_JET_LOOSE_WP = "VVVLoose"  # lower edge of the anti-isolated region
TAU_VS_MU_WP = {"mt": "Tight", "et": "VLoose", "tt": "VLoose"}
TAU_VS_ELE_WP = {"mt": "VVLoose", "et": "Tight", "tt": "VVLoose"}

# b-tag multiplicity bins of the control regions (each combined with >= 2 jets)
BTAG_BINS = {"btag0": "n_bjets == 0", "btag1": "n_bjets == 1", "btag2": "n_bjets == 2", "btag3p": "n_bjets >= 3"}

# NN output classes in training order: index = friend column predicted_class, datacard bin id = index + 1
NN_CLASS_NAMES = ("HH2B2Tau", "DY", "ST", "TT", "VV", "jetFakes", "Other")
NN_CLASS_COLUMN = "predicted_class"
NN_SCORE_COLUMN = "predicted_max_value"
NN_COLUMNS = frozenset((NN_CLASS_COLUMN, NN_SCORE_COLUMN))

# fake-factor friend (CROWN fake_factors_friend_config.py); verify against the first produced friend file
FF_COLUMNS = {"lt": "fake_factor", "tt_1": "fake_factor_1", "tt_2": "fake_factor_2"}
FF_COLUMN_SET = frozenset(FF_COLUMNS.values())

# embedding ntuple columns (unverified until 2018 v15 embedding ntuples exist)
EMBEDDING_COLUMNS = frozenset(("emb_genweight", "emb_idsel_wgt_1", "emb_idsel_wgt_2", "emb_triggersel_wgt"))

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
