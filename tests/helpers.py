import functools
import json
from pathlib import Path

from shapesmith.config import RunConfig
from shapesmith.samples import read_sample_list

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "inventory"
FIXTURES = REPO / "tests" / "fixtures"
DATABASE = FIXTURES / "datasets.json"
BRANCHES = FIXTURES / "branches"  # <era>.json, written by scripts/branches_from_crown.py
EMBEDDING_LIST = "sm2018_embedding"
ERAS = ("2016preVFP", "2016postVFP", "2017", "2018")
YEARS = {"2016preVFP": "2016", "2016postVFP": "2016", "2017": "2017", "2018": "2018"}

# Part-A output contract of the embedded samples (docs/superpowers/specs/2026-09-29-sm-embedding-2018-design.md §7);
# the per-era fixtures contain them
EMBEDDING_EXTRA = {"emb_genweight", "emb_idsel_wgt_1", "emb_idsel_wgt_2", "emb_triggersel_wgt"}
EMBEDDING_EXTRA_ET = {"iso_wgt_ele_1"}
# CROWN output of the v6 production (B7) that the diagnostics read; the per-era fixtures contain them
V6_COLUMNS = {"npv_good", "rho"}


@functools.cache
def _era_fixture(era: str) -> dict:
    return json.loads((BRANCHES / f"{era}.json").read_text())


def fixture_branches(channel: str, era: str = "2018", sample_type: str = "ttbar", config: str = "sm_config") -> set[str]:
    """The branches CROWN's generated program of the configuration, era, sample type and scope writes (the fixture
    tests/fixtures/branches/<era>.json, from the CROWN dry run of bbtautau feat-v7-run2; regenerate with
    scripts/branches_from_crown.py, or from real files with scripts/dump_branches.py)."""
    scopes = _era_fixture(era)["configs"][config]
    entry = scopes[channel]
    base = set(entry["ttbar"])
    if sample_type == "ttbar":
        return base
    if sample_type not in entry:
        raise KeyError(f"{config} {era}: no {sample_type} program for {channel}")
    delta = entry[sample_type]
    return (base - set(delta["remove"])) | set(delta["add"])


def kind_of(sample_type: str) -> str:
    return sample_type if sample_type in ("data", "embedding") else "mc"


# The v7 CROWN output that the branch of the fixtures does not write yet (spec C6-C8, C11, C13, C14, C20; tt embedding
# E33), the counterpart of the PENDING items of the CROWN tests: per era the trigger flags (every kind) and scale
# factors (MC, embedding) of its triggers in place of the fixture's ones, the prefiring weight with its Up/Dn columns
# (MC), the fixed-WP [T, M] b-tag weight variations instead of the 40 shape components (MC), the HEM flag of 2018 (every
# kind), the tt embedding's di-tau data efficiencies, and the trigger monitoring columns of the tau-ID production.
# Delete an entry once the fixture comes from a CROWN branch that writes it.
V7_TRIGGER_COLUMNS = {
    "2016": {
        "muon": {"trg_single_mu22", "trg_single_mu22_tk", "trg_single_mu22_eta2p1", "trg_single_mu22_eta2p1_tk", "trg_wgt_single_mu22"},
        "electron": {"trg_single_ele25", "trg_wgt_single_ele25"},
        "tt": {"trg_double_tau35_mediumiso", "trg_double_tau35_mediumcombinediso",
               "trg_wgtdouble_tau35_mediumiso_leg1", "trg_wgtdouble_tau35_mediumiso_leg2",
               "trg_wgtdouble_tau35_mediumcombinediso_leg1", "trg_wgtdouble_tau35_mediumcombinediso_leg2"},
        "tau_id": {"trg_single_mu22", "trg_single_mu22_tk", "trg_single_mu22_eta2p1", "trg_single_mu22_eta2p1_tk", "trg_wgt_single_mu22"},
    },
    "2017": {
        "muon": {"trg_single_mu27", "trg_wgt_single_mu27"},
        "electron": {"trg_single_ele32", "trg_single_ele35", "trg_wgt_single_ele32", "trg_wgt_single_ele35"},
        "tt": {f"trg_{path}" for path in ("double_tau35_tightiso", "double_tau40_mediumiso", "double_tau40_tightiso")}
        | {f"trg_wgt{path}_leg{leg}" for path in ("double_tau35_tightiso", "double_tau40_mediumiso", "double_tau40_tightiso") for leg in (1, 2)},
        "tau_id": {"trg_single_mu27", "trg_wgt_single_mu27"},
    },
    "2018": {
        "muon": {"trg_single_mu24", "trg_wgt_single_mu24"},
        "electron": {"trg_single_ele32", "trg_wgt_single_ele32"},
        "tt": {f"trg_{path}" for path in ("double_tau35_mediumiso", "double_tau35_tightiso", "double_tau40_mediumiso", "double_tau40_tightiso")}
        | {f"trg_wgt{path}_leg{leg}" for path in ("double_tau35_mediumiso", "double_tau35_tightiso", "double_tau40_mediumiso", "double_tau40_tightiso") for leg in (1, 2)},
        "tau_id": {"trg_single_mu24", "trg_single_mu27", "trg_wgt_single_mu24", "trg_wgt_single_mu27"},
    },
}
V7_TRIGGER_LEPTONS = {"mt": "muon", "em": "muon", "mm": "muon", "et": "electron", "ee": "electron", "tt": "tt"}
V7_BTAG = {f"btag_weight_upart_{d}_{flavour}_{correlation}" for d in ("up", "down") for flavour in ("bc", "light") for correlation in ("correlated", "uncorrelated")}
V7_PREFIRING = {"prefiring_wgt", "L1PreFiringWeight_Up", "L1PreFiringWeight_Dn"}
V7_EMBEDDING_DITAU = {f"trg_effdata_emb_{wp}_{leg}" for wp in ("Medium", "AntiMedium") for leg in (1, 2)}
# C13 / v7_tt_emb_trigger 12.4: the trigger monitoring columns of the tau-ID production (mt, every kind)
V7_TAU_ID_MONITOR = {
    "2016": {"trg_mon_mu19tau32_mediso_2", "trg_mon_mu19tau32_medcombiso_2"},
    "2017": {"trg_mon_mu24tau35_tightiso_tightid_2", "trg_mon_mu24tau35_mediumiso_tightid_2", "trg_mon_mu24tau35_tightiso_2"},
    "2018": {"trg_mon_mu24tau35_tightiso_tightid_2", "trg_mon_mu24tau35_mediumiso_tightid_2", "trg_mon_mu24tau35_tightiso_2", "trg_mon_mu24tau35_mediumiso_hps_2"},
}
V7_TRIGGER_OBJECT = {"trgobj_pt_2", "trgobj_l1pt_2"}


def _triggers(columns: set[str], kind: str) -> set[str]:
    """Trigger flags for every kind, scale factors for MC and embedding."""
    return columns if kind != "data" else {c for c in columns if not c.startswith("trg_wgt")}


def branches(channel: str, era: str = "2018", sample_type: str = "ttbar") -> set[str]:
    """The branches of a v7 sm_config n-tuple of the channel, era and sample type: the fixture plus the pending v7
    CROWN output (V7_* above)."""
    kind = kind_of(sample_type)
    base = {b for b in fixture_branches(channel, era, sample_type) if not b.startswith("trg_") and not b.startswith("btag_weight_upart_")}
    added = _triggers(V7_TRIGGER_COLUMNS[YEARS[era]][V7_TRIGGER_LEPTONS[channel]], kind) | ({"hem_veto"} if era == "2018" else set())
    if kind == "mc":
        added |= V7_BTAG | V7_PREFIRING
    if kind == "embedding" and channel == "tt":
        added |= V7_EMBEDDING_DITAU
    return base | added


def embedding_branches(channel: str, era: str = "2018") -> set[str]:
    """The branches of a v7 embedding n-tuple of a tau channel (tt: with the di-tau data efficiencies of both isolation
    classes, E33)."""
    return branches(channel, era, "embedding")


def tau_id_branches(channel: str, era: str = "2018", sample_type: str = "ttbar") -> set[str]:
    """The branches of a v7 n-tuple of the tau-ID production (sm_tau_id_measurement_config, mt and mm): the fixture plus
    the pending output (the muon triggers of the era, C6/C13; in mt the monitoring columns, 12.4; prefiring, C11; HEM)."""
    kind = kind_of(sample_type)
    base = {b for b in fixture_branches(channel, era, sample_type, "sm_tau_id_measurement_config") if not b.startswith("trg_")}
    added = _triggers(V7_TRIGGER_COLUMNS[YEARS[era]]["tau_id"], kind) | ({"hem_veto"} if era == "2018" else set())
    if channel == "mt":
        added |= V7_TAU_ID_MONITOR[YEARS[era]] | V7_TRIGGER_OBJECT
    if kind == "mc":
        added |= V7_PREFIRING
    return base | added


@functools.cache
def sample_types() -> dict[str, str]:
    """nick -> sample_type of the fixture database."""
    return {nick: entry["sample_type"] for nick, entry in json.loads(DATABASE.read_text()).items()}


def embedded(embedding: bool | list[str], channel: str) -> bool:
    """The switch embedding (true/false or the embedded channels) for one channel."""
    return channel in embedding if isinstance(embedding, list) else embedding


def nicks(sample_list: str = "sm2018_binned_v2") -> tuple[str, ...]:
    return read_sample_list(INVENTORY / f"{sample_list}.txt")


TAU_ANALYSIS = "bbtautau_shapesmith.analysis:build"


def config(channels=("et", "mt", "tt"), analysis=TAU_ANALYSIS, era="2018", database=DATABASE, **switches) -> RunConfig:
    """A run configuration on the fixture database; the sample lists default to the v2 production. A tau run without
    the NN friend has no fit categories unless it names them (fit_categories nn needs the friend, NN spec X9)."""
    switches.setdefault("sample_lists", ["sm2018_binned_v2"])
    if analysis == TAU_ANALYSIS:
        switches.setdefault("fit_categories", "nn" if switches.get("nn_friend") else "none")
    return RunConfig(analysis=analysis, era=era, channels=list(channels), switches=switches, ntuples={"base": "/unused"},
                     skim_dir="/unused/skim", output_dir="/unused/output", sample_database=database)
