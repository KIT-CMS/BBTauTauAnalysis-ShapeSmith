"""The README and the comments of the run YAMLs name only the shapesmith commands and options of the CLI
(shapesmith.readme_check) and only switches of the analyses (`switches.<name>`)."""
import re

import pytest
from shapesmith.cli import app
from shapesmith.readme_check import problems

from bbtautau_shapesmith.switches import DileptonSwitches, FakeFactorSwitches, TauIdSwitches, TauSwitches, TauTriggerSwitches
from tests.helpers import REPO

SWITCH = re.compile(r"\bswitches\.(?!py\b)([A-Za-z_0-9]+)")  # not the module switches.py
SWITCHES = set().union(*(model.model_fields for model in (TauSwitches, FakeFactorSwitches, DileptonSwitches, TauIdSwitches, TauTriggerSwitches)))
CONFIGS = sorted((REPO / "configs").glob("*.yaml"))


def _comments(path) -> str:
    """The comments of a run YAML as one code block (their commands)."""
    return "```\n" + "\n".join(line.split("#", 1)[1] for line in path.read_text().splitlines() if "#" in line) + "\n```\n"


def test_the_readme_names_only_commands_and_options_of_the_cli():
    assert problems((REPO / "README.md").read_text(), app) == []


@pytest.mark.parametrize("path", CONFIGS, ids=[path.stem for path in CONFIGS])
def test_the_run_yaml_comments_name_only_commands_and_options_of_the_cli(path):
    assert problems(_comments(path), app) == []


def test_the_readme_and_the_run_yamls_name_only_existing_switches():
    texts = {"README.md": (REPO / "README.md").read_text(), **{path.name: path.read_text() for path in CONFIGS}}
    named = {name for text in texts.values() for name in SWITCH.findall(text)}
    assert {"nn_prefix", "signal_sample", "heavy_flavour"} <= named
    assert {f"{file}: switches.{name}" for file, text in texts.items() for name in SWITCH.findall(text) if name not in SWITCHES} == set()
