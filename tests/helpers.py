from pathlib import Path

from shapesmith.config import RunConfig
from shapesmith.samples import read_sample_list

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "inventory"
FIXTURES = REPO / "tests" / "fixtures"
DATABASE = FIXTURES / "datasets.json"
# embedding ntuple columns (unverified until 2018 v15 embedding ntuples exist)
EMBEDDING_COLUMNS = {"emb_genweight", "emb_idsel_wgt_1", "emb_idsel_wgt_2", "emb_triggersel_wgt"}


def branches(channel: str) -> set[str]:
    """Branch names of a CROWN sm_config ttbar ntuple of the channel (regenerate with scripts/dump_branches.py)."""
    return {line.strip() for line in (FIXTURES / f"branches_{channel}.txt").read_text().splitlines() if line.strip()}


def nicks(sample_list: str = "sm2018_binned_v2") -> tuple[str, ...]:
    return read_sample_list(INVENTORY / f"{sample_list}.txt")


def config(channels=("et", "mt", "tt"), analysis="bbtautau_shapesmith.analysis:build", **switches) -> RunConfig:
    """A run configuration on the fixture database; the sample lists default to the v2 production."""
    switches.setdefault("sample_lists", ["sm2018_binned_v2"])
    return RunConfig(analysis=analysis, era="2018", channels=list(channels), switches=switches, ntuples={"base": "/unused"},
                     skim_dir="/unused/skim", output_dir="/unused/output", sample_database=DATABASE)
