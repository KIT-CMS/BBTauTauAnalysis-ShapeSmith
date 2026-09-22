from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "inventory"
FIXTURES = REPO / "tests" / "fixtures"
DATABASE = FIXTURES / "datasets.json"


def branches(channel: str) -> set[str]:
    """Branch names of a CROWN sm_config ttbar ntuple of the channel (regenerate with scripts/dump_branches.py)."""
    return {line.strip() for line in (FIXTURES / f"branches_{channel}.txt").read_text().splitlines() if line.strip()}


def nicks(sample_list: str = "sm2018_binned_v2") -> list[str]:
    return [line.split()[0] for line in (INVENTORY / f"{sample_list}.txt").read_text().splitlines() if line.strip()]
