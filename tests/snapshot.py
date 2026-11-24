"""The content of the Analysis that every run YAML builds, in a canonical form and as a digest: what a run declares
(samples, selections, processes, regions, categories, variables, variations, estimators, lnN, ML export). The digests
of tests/fixtures/run_snapshot.json pin the runs (its `state` says on which commits): tests/test_run_snapshot.py
compares every run with them, so that a change to the model that is meant for new runs only cannot move an old one
unnoticed. After an intended change of an old run: scripts/run_snapshot.py --write.

The module depends on the installed packages only through shapesmith.config, so that it also runs on an older state
of both (PYTHONPATH on their sources and on this directory's parent). Fields that a later state added to the model
are left out where they keep their default (DEFAULTS): an old and a new state give the same content for the same run.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
DATABASE = REPO / "tests" / "fixtures" / "datasets.json"
FIXTURE = REPO / "tests" / "fixtures" / "run_snapshot.json"
ERAS = ("2016preVFP", "2016postVFP", "2017", "2018")
YEARS = ("2016", "2017", "2018")  # the eras of the fake-factor measurements of Run 2 (one of 2016)
# switches naming files outside the repository (payloads, model manifests, cards of other runs): unset for the snapshot
EXTERNAL = ("nn_manifest", "nn_blinding", "nn_ff_payload", "emb_es_payload", "reference_payload", "kit_reference_payload")
SKIPPED = ("style", "measurement")  # plot cosmetics and the measurement objects (callables, paths)
# model fields added after the pinned state -> their default: left out of the content while they have it
DEFAULTS = {"ColumnVariation": {"required": ()}, "VariationSum": {"scale": 1.0, "source": None}}
ADDRESS = re.compile(r" at 0x[0-9a-f]+")


def eras_of(name: str) -> tuple[str, ...]:
    """The eras a run YAML is built for: 2018 for the 2018 productions, the measurement years for the fake-factor
    measurements of Run 2, else the four eras."""
    if not name.startswith(("smrun2", "ff_smrun2")):
        return ("2018",)
    return YEARS if name.startswith("ff_") else ERAS


def runs() -> list[tuple[str, str]]:
    """(run YAML, era) of every run."""
    return [(path.stem, era) for path in sorted(CONFIGS.glob("*.yaml")) for era in eras_of(path.stem)]


def content(value):
    """A JSON value of an analysis object: dataclasses by field, mappings by sorted key, sequences as lists."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        fields = {f.name: getattr(value, f.name) for f in dataclasses.fields(value) if f.name not in SKIPPED}
        fields = {key: item for key, item in fields.items() if not (key in DEFAULTS.get(name, {}) and item == DEFAULTS[name][key])}
        return {"type": name, **{key: content(item) for key, item in fields.items()}}
    if isinstance(value, Mapping):
        return {str(key): content(item) for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))}
    if isinstance(value, (list, tuple)):
        return [content(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return ADDRESS.sub("", repr(value))


def switches_of(name: str) -> dict:
    """The switches a run YAML sets."""
    return (yaml.safe_load((CONFIGS / f"{name}.yaml").read_text()) or {}).get("switches", {})


def build(name: str, era: str, *overrides: str):
    """The Analysis of a run YAML in an era on the fixture database, without the files outside the repository;
    `overrides` as `-s key=value`."""
    from shapesmith.config import load_analysis, load_config

    external = [f"switches.{switch}=null" for switch in EXTERNAL if switches_of(name).get(switch) is not None]
    return load_analysis(load_config(CONFIGS / f"{name}.yaml", [f"era={era}", f"sample_database={DATABASE}", *external, *overrides]))


def run_content(name: str, era: str, *overrides: str) -> dict:
    """The content of a run, or the error its build raises (a run that an era lacks inputs for)."""
    try:
        return content(build(name, era, *overrides))
    except Exception as error:  # the kind of failure is part of the pinned behaviour
        return {"error": type(error).__name__, "message": ADDRESS.sub("", str(error)).replace(str(REPO), "<repo>")}


def without_declared_only(run: dict) -> dict:
    """The content of a run without its declared-only column variations (regions = (): never read or filled), which
    change nothing a run makes."""
    channels = {
        name: {**channel, "variations": [v for v in channel["variations"] if not (v["type"] == "ColumnVariation" and v["regions"] == [])]}
        for name, channel in run.get("channels", {}).items()
    }
    return {**run, "channels": channels} if channels else run


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def snapshot() -> dict[str, dict]:
    """`<run>/<era>` -> the content of every run."""
    return {f"{name}/{era}": run_content(name, era) for name, era in runs()}


def differences(old, new, path: str = "") -> list[str]:
    """Where two contents differ, as `<path>: <old> -> <new>`; lists of named objects are compared by name."""
    if isinstance(old, dict) and isinstance(new, dict):
        return [line for key in sorted(set(old) | set(new)) for line in differences(old.get(key, "<absent>"), new.get(key, "<absent>"), f"{path}/{key}")]
    if isinstance(old, list) and isinstance(new, list):
        if all(isinstance(item, dict) and "name" in item for item in old + new):
            def keyed(items):
                seen, result = {}, {}
                for item in items:
                    seen[item["name"]] = seen.get(item["name"], 0) + 1
                    result[item["name"] if seen[item["name"]] == 1 else f"{item['name']}#{seen[item['name']]}"] = item
                return result
            first, second = keyed(old), keyed(new)
            order = [] if list(first) == list(second) or set(first) != set(second) else [f"{path}: order {list(first)[:5]}... -> {list(second)[:5]}..."]
            return order + [line for key in dict.fromkeys([*first, *second]) for line in differences(first.get(key, "<absent>"), second.get(key, "<absent>"), f"{path}/{key}")]
        if len(old) == len(new):
            return [line for index, (a, b) in enumerate(zip(old, new)) for line in differences(a, b, f"{path}[{index}]")]
    return [] if old == new else [f"{path}: {json.dumps(old)[:200]} -> {json.dumps(new)[:200]}"]
