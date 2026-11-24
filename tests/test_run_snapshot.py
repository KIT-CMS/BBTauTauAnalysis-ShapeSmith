"""Every run YAML builds the Analysis it built when it was pinned (tests/snapshot.py: the digests of
tests/fixtures/run_snapshot.json): a change of the model that is meant for new runs cannot move an old one unnoticed.
After an intended change of an old run: scripts/run_snapshot.py --write."""
import json

import pytest

from tests import snapshot

PINNED = json.loads(snapshot.FIXTURE.read_text())
RUNS = snapshot.runs()


def test_every_run_yaml_is_pinned():
    assert sorted(PINNED["runs"]) == sorted(f"{name}/{era}" for name, era in RUNS) and len(RUNS) >= 125
    assert PINNED["state"]  # what the digests pin, e.g. the commits of both repositories


@pytest.mark.parametrize("name,era", RUNS, ids=[f"{name}-{era}" for name, era in RUNS])
def test_the_run_builds_what_it_built_before(name, era):
    """Samples, selections, processes, regions, categories, variables, variations, estimators, lnN and ML export."""
    assert snapshot.digest(snapshot.run_content(name, era)) == PINNED["runs"][f"{name}/{era}"], "scripts/run_snapshot.py --dump and --compare show what moved"
