from pathlib import Path

from shapesmith.config import RunConfig
from shapesmith.samples import read_sample_list

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "inventory"
FIXTURES = REPO / "tests" / "fixtures"
DATABASE = FIXTURES / "datasets.json"
EMBEDDING_LIST = "sm2018_embedding"

# Part-A output contract of the embedded samples (docs/superpowers/specs/2026-09-29-sm-embedding-2018-design.md §7),
# relative to the MC branch fixtures, until a branch fixture of a real embedding file exists.
EMBEDDING_EXTRA = {"emb_genweight", "emb_idsel_wgt_1", "emb_idsel_wgt_2", "emb_triggersel_wgt"}
EMBEDDING_EXTRA_ET = {"iso_wgt_ele_1"}
# CROWN output of the v6 production (B7) that the diagnostics read, until a branch fixture of a v6 file exists
V6_COLUMNS = {"npv_good", "rho"}


def fixture_branches(channel: str) -> set[str]:
    """Branch names of a CROWN sm_config ttbar ntuple of the channel of 2018 (sm2018_binned_v2; regenerate with
    scripts/dump_branches.py)."""
    return {line.strip() for line in (FIXTURES / f"branches_{channel}.txt").read_text().splitlines() if line.strip()}


# The v7 CROWN output (v7 spec C6-C8, C11, C14, C20; tt embedding 12.6) relative to the 2018 fixtures, until branch
# fixtures of v7 files exist: per era the trigger flags and scale factors of its triggers (the 2018 fixture's ones
# removed), the prefiring weight with its Up/Dn columns (MC), the fixed-WP [T, M] b-tag weight variations instead of the
# 40 shape components, and the HEM flag of 2018.
V7_TRIGGER_COLUMNS = {
    "2016": {
        "muon": {"trg_single_mu22", "trg_single_mu22_tk", "trg_single_mu22_eta2p1", "trg_single_mu22_eta2p1_tk", "trg_wgt_single_mu22"},
        "electron": {"trg_single_ele25", "trg_wgt_single_ele25"},
        "tt": {"trg_double_tau35_mediumiso", "trg_double_tau35_mediumcombinediso",
               "trg_wgtdouble_tau35_mediumiso_leg1", "trg_wgtdouble_tau35_mediumiso_leg2",
               "trg_wgtdouble_tau35_mediumcombinediso_leg1", "trg_wgtdouble_tau35_mediumcombinediso_leg2"},
    },
    "2017": {
        "muon": {"trg_single_mu27", "trg_wgt_single_mu27"},
        "electron": {"trg_single_ele32", "trg_single_ele35", "trg_wgt_single_ele32", "trg_wgt_single_ele35"},
        "tt": {f"trg_{path}" for path in ("double_tau35_tightiso", "double_tau40_mediumiso", "double_tau40_tightiso")}
        | {f"trg_wgt{path}_leg{leg}" for path in ("double_tau35_tightiso", "double_tau40_mediumiso", "double_tau40_tightiso") for leg in (1, 2)},
    },
    "2018": {
        "muon": {"trg_single_mu24", "trg_wgt_single_mu24"},
        "electron": {"trg_single_ele32", "trg_wgt_single_ele32"},
        "tt": {f"trg_{path}" for path in ("double_tau35_mediumiso", "double_tau35_tightiso", "double_tau40_mediumiso", "double_tau40_tightiso")}
        | {f"trg_wgt{path}_leg{leg}" for path in ("double_tau35_mediumiso", "double_tau35_tightiso", "double_tau40_mediumiso", "double_tau40_tightiso") for leg in (1, 2)},
    },
}
V7_TRIGGER_LEPTONS = {"mt": "muon", "em": "muon", "mm": "muon", "et": "electron", "ee": "electron", "tt": "tt"}
V7_BTAG = {f"btag_weight_upart_{d}_{flavour}_{correlation}" for d in ("up", "down") for flavour in ("bc", "light") for correlation in ("correlated", "uncorrelated")}
V7_PREFIRING = {"prefiring_wgt", "L1PreFiringWeight_Up", "L1PreFiringWeight_Dn"}
V7_EMBEDDING_DITAU = {f"trg_effdata_emb_{wp}_{leg}" for wp in ("Medium", "AntiMedium") for leg in (1, 2)}
YEARS = {"2016preVFP": "2016", "2016postVFP": "2016", "2017": "2017", "2018": "2018"}


def branches(channel: str, era: str = "2018") -> set[str]:
    """The branches of a v7 sm_config MC n-tuple of the channel and era (the v7 contract on the 2018 fixture)."""
    base = {b for b in fixture_branches(channel) if not b.startswith("trg_") and not b.startswith("btag_weight_upart_")}
    added = V7_TRIGGER_COLUMNS[YEARS[era]][V7_TRIGGER_LEPTONS[channel]] | V7_BTAG | V7_PREFIRING | ({"hem_veto"} if era == "2018" else set())
    return base | added


def embedding_branches(channel: str, era: str = "2018") -> set[str]:
    """The branches an embedding file of the channel has by the Part-A contract: the MC ones without pileup, prefiring,
    b-tag and generator weights, plus the embedding weights (tt: the di-tau data efficiencies of both isolation
    classes, E33)."""
    mc = branches(channel, era)
    absent = {b for b in mc if b.startswith("btag_weight_upart")} | V7_PREFIRING | {"puweight", "genWeight", "reco_wgt_ele_1"}
    extra = EMBEDDING_EXTRA | (EMBEDDING_EXTRA_ET if channel == "et" else set()) | (V7_EMBEDDING_DITAU if channel == "tt" else set())
    return (mc - absent) | extra


def embedded(embedding: bool | list[str], channel: str) -> bool:
    """The switch embedding (true/false or the embedded channels) for one channel."""
    return channel in embedding if isinstance(embedding, list) else embedding


def nicks(sample_list: str = "sm2018_binned_v2") -> tuple[str, ...]:
    return read_sample_list(INVENTORY / f"{sample_list}.txt")


def config(channels=("et", "mt", "tt"), analysis="bbtautau_shapesmith.analysis:build", era="2018", database=DATABASE, **switches) -> RunConfig:
    """A run configuration on the fixture database; the sample lists default to the v2 production."""
    switches.setdefault("sample_lists", ["sm2018_binned_v2"])
    return RunConfig(analysis=analysis, era=era, channels=list(channels), switches=switches, ntuples={"base": "/unused"},
                     skim_dir="/unused/skim", output_dir="/unused/output", sample_database=database)
