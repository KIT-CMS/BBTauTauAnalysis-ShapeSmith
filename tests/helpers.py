from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "inventory"
DATABASE = REPO / "tests" / "fixtures" / "datasets.json"


def inventory(channel: str) -> set[str]:
    return {line.strip() for line in (INVENTORY / f"2018_v15_{channel}.txt").read_text().splitlines() if line.strip()}


def nicks(sample_list: str = "sm2018_binned_v2") -> list[str]:
    return [line.split()[0] for line in (INVENTORY / f"{sample_list}.txt").read_text().splitlines() if line.strip()]
