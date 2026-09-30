"""The stored sm2018_binned_v4 skims stay reusable: their contracts and Parquet columns (one data and one MC sample per
channel, cut from the golden references of 2026-09-30) against the contracts the v4 run YAMLs ask for today."""
import json

import pytest
from shapesmith.config import load_analysis, load_config
from shapesmith.skim import contract, needed_columns, reuse_problems

from tests.helpers import DATABASE, FIXTURES, REPO

STORED = json.loads((FIXTURES / "skim_contracts_v4.json").read_text())
RUN_YAML = {"et": "sm2018_binned_v4", "mt": "sm2018_binned_v4", "tt": "sm2018_binned_v4",
            "em": "sm2018_binned_v4_dilepton", "mm": "sm2018_binned_v4_dilepton", "ee": "sm2018_binned_v4_dilepton"}


@pytest.mark.parametrize("channel_name", sorted(STORED))
def test_v4_skims_are_reused(channel_name):
    config = load_config(REPO / "configs" / f"{RUN_YAML[channel_name]}.yaml").model_copy(update={"sample_database": DATABASE})
    channel = load_analysis(config).channel(channel_name)
    stored = STORED[channel_name]
    assert dict(channel.skim) == stored["selection"]  # byte for byte
    samples = {s.nick: s for s in channel.samples}
    for nick, skim in stored["samples"].items():
        expected = contract(config, channel, samples[nick], needed_columns(channel, samples[nick]))
        assert reuse_problems(skim["contract"], expected, set(skim["parquet_columns"])) == [], nick
