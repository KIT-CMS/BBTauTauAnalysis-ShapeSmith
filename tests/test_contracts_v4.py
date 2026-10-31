"""The stored sm2018_binned_v4 skims against the contracts the v4 run YAMLs ask for today (one data and one MC sample per
channel, cut from the golden references of 2026-09-30): the only differences are the changes since, so a v4 skim is
not reused, and why."""
import json

import pytest
from shapesmith.config import load_analysis, load_config
from shapesmith.skim import contract, needed_columns, reuse_problems

from tests.helpers import DATABASE, FIXTURES, REPO, V7_BTAG, V7_PREFIRING

STORED = json.loads((FIXTURES / "skim_contracts_v4.json").read_text())
RUN_YAML = {"et": "sm2018_binned_v4", "mt": "sm2018_binned_v4", "tt": "sm2018_binned_v4",
            "em": "sm2018_binned_v4_dilepton", "mm": "sm2018_binned_v4_dilepton", "ee": "sm2018_binned_v4_dilepton"}
TT_TRIGGER = ("((is_embedding > 0.5) | ((trg_double_tau35_mediumiso > 0.5) | (trg_double_tau35_tightiso > 0.5) | (trg_double_tau40_mediumiso > 0.5)"
              " | (trg_double_tau40_tightiso > 0.5)))")


def selection_changes(channel_name: str) -> dict[str, str]:
    """The skim cuts added or changed since v4: the tau eta cut in et and mt (v6, A5); v7: the HEM veto of 2018 (E29),
    the light-lepton eta in et and mt (E31), the tt trigger cut without requirement for the embedded sample and the
    tt tau eta of the trigger acceptance (E33, E7)."""
    changes = {"hem_veto": "(hem_veto < 0.5)"}
    if channel_name in ("et", "mt"):
        changes |= {"tau_eta": "(abs(eta_2) < 2.3)", "lepton_eta": "(abs(eta_1) < 2.1)"}
    if channel_name == "tt":
        changes |= {"trg_selection": TT_TRIGGER, "tau_eta": "(abs(eta_1) < 2.1) & (abs(eta_2) < 2.1)"}
    return changes


def new_columns(channel_name: str, kind: str) -> set[str]:
    """The columns read since v4: the HEM flag; on MC the prefiring weight and its Up/Dn columns (E11, C11) and the
    fixed-WP b-tag variations (C14), in et the electron reco SF (v6, A3)."""
    if kind == "data":
        return {"hem_veto"}
    return {"hem_veto"} | V7_PREFIRING | V7_BTAG | ({"reco_wgt_ele_1"} if channel_name == "et" else set())


@pytest.mark.parametrize("channel_name", sorted(STORED))
def test_v4_skims_differ_by_the_changes_since(channel_name):
    config = load_config(REPO / "configs" / f"{RUN_YAML[channel_name]}.yaml").model_copy(update={"sample_database": DATABASE})
    channel = load_analysis(config).channel(channel_name)
    stored = STORED[channel_name]
    assert dict(channel.skim) == stored["selection"] | selection_changes(channel_name)  # byte for byte
    samples = {s.nick: s for s in channel.samples}
    for nick, skim in stored["samples"].items():
        expected = contract(config, channel, samples[nick], needed_columns(channel, samples[nick]))
        assert set(expected["required_columns"]) - set(skim["parquet_columns"]) == new_columns(channel_name, samples[nick].kind), nick
        assert [p for p in reuse_problems(skim["contract"], expected, set(skim["parquet_columns"])) if not p.startswith("columns missing")] == ["selection changed"], nick
