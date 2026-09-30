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


def branches(channel: str) -> set[str]:
    """Branch names of a CROWN sm_config ttbar ntuple of the channel (regenerate with scripts/dump_branches.py)."""
    return {line.strip() for line in (FIXTURES / f"branches_{channel}.txt").read_text().splitlines() if line.strip()}


def embedding_branches(channel: str) -> set[str]:
    """The branches an embedding file of the channel has by the Part-A contract: the MC ones without pileup, b-tag and
    generator weights, plus the embedding weights."""
    absent = {b for b in branches(channel) if b.startswith("btag_weight_upart")} | {"puweight", "genWeight", "reco_wgt_ele_1"}
    return (branches(channel) - absent) | EMBEDDING_EXTRA | (EMBEDDING_EXTRA_ET if channel == "et" else set())


def nicks(sample_list: str = "sm2018_binned_v2") -> tuple[str, ...]:
    return read_sample_list(INVENTORY / f"{sample_list}.txt")


def config(channels=("et", "mt", "tt"), analysis="bbtautau_shapesmith.analysis:build", **switches) -> RunConfig:
    """A run configuration on the fixture database; the sample lists default to the v2 production."""
    switches.setdefault("sample_lists", ["sm2018_binned_v2"])
    return RunConfig(analysis=analysis, era="2018", channels=list(channels), switches=switches, ntuples={"base": "/unused"},
                     skim_dir="/unused/skim", output_dir="/unused/output", sample_database=DATABASE)
