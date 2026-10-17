#!/usr/bin/env python3
"""Write the per-era branch fixtures tests/fixtures/branches/<era>.json from CROWN's generated code.

CROWN's generate.py writes one C++ program per configuration, era, sample type and scopes; its
`output_quantities` block names every branch the program writes per scope. This script reads them for the
configurations the analysis runs on (sm_config: one sample type and scope per program; sm_tau_id_measurement_config:
the scopes mt and mm together) and stores, per era, configuration and scope, the branches of the ttbar program in
full and those of every other sample type as the difference to it:

    {"era": ..., "source": ..., "fallback": [surfaces read from a later --generated directory],
     "configs": {"sm_config": {"mt": {"ttbar": [...], "dyjets": {"add": [...], "remove": [...]}, ...}}, ...}}

Layout of a --generated directory (the dry run of the CROWN tests, run_matrix.py):
<dir>/<config>/<era>/<sample type>/<scopes>/<name>_generated_code/<name>.cxx. Several --generated directories are read
in order, the first one with a program wins (e.g. the dry run of the branch, then a patched copy for the programs the
branch cannot generate yet). The fixtures of real files: scripts/dump_branches.py.

    python scripts/branches_from_crown.py --generated DRYRUN/new/gen --generated DRYRUN/patch/gen \\
        --source "CROWN bbtautau feat-v7-run2 cfcca94 ..." --output tests/fixtures/branches
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ERAS = ("2016preVFP", "2016postVFP", "2017", "2018")
BASE_TYPE = "ttbar"
CONFIGS = {  # configuration -> scope -> the scopes directory of its program
    "sm_config": {scope: scope for scope in ("et", "mt", "tt", "em", "ee", "mm")},
    "sm_tau_id_measurement_config": {"mt": "mt-mm", "mm": "mt-mm"},
}
OUTPUTS = re.compile(r"output_quantities = \{(.*?)\};", re.S)
SCOPE_BLOCK = re.compile(r"\{\s*(\w+),\s*\{(.*?)\}\s*\}", re.S)
LITERAL = re.compile(r'"([^"]+)"')


def outputs(program: Path) -> dict[str, set[str]]:
    """Scope -> the branches of a generated program."""
    result: dict[str, set[str]] = {}
    for block in OUTPUTS.findall(program.read_text()):
        for path, columns in SCOPE_BLOCK.findall(block):
            result.setdefault(path.removeprefix("outputpath_"), set()).update(LITERAL.findall(columns))
    return result


def program(directory: Path) -> Path | None:
    found = sorted(directory.glob("*_generated_code/*.cxx"))
    return found[0] if found else None


def surface(generated: list[Path], config: str, era: str, sample_type: str, scopes: str) -> tuple[dict[str, set[str]], int] | None:
    """The outputs of the first --generated directory that has the program, and its index."""
    for index, root in enumerate(generated):
        path = program(root / config / era / sample_type / scopes)
        if path is not None:
            return outputs(path), index
    return None


def era_fixture(generated: list[Path], era: str, source: str) -> dict:
    configs, fallback = {}, []
    for config, scope_dirs in CONFIGS.items():
        types = sorted({path.name for root in generated for path in (root / config / era).glob("*") if path.is_dir()})
        if BASE_TYPE not in types:
            raise SystemExit(f"{config} {era}: no {BASE_TYPE} program in {', '.join(map(str, generated))}")
        per_scope: dict[str, dict] = {}
        for scope, scopes in scope_dirs.items():
            found = {}
            for sample_type in types:
                result = surface(generated, config, era, sample_type, scopes)
                if result is None or scope not in result[0]:
                    continue
                found[sample_type] = result[0][scope]
                if result[1]:
                    fallback.append(f"{config}/{sample_type}/{scopes}")
            base = found.pop(BASE_TYPE)
            per_scope[scope] = {BASE_TYPE: sorted(base)} | {
                sample_type: {"add": sorted(columns - base), "remove": sorted(base - columns)} for sample_type, columns in sorted(found.items())
            }
        configs[config] = per_scope
    return {"era": era, "source": source, "fallback": sorted(set(fallback)), "configs": configs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--generated", type=Path, action="append", required=True, help="directory of generated programs (repeat: fallbacks in order)")
    parser.add_argument("--source", required=True, help="what generated the programs (CROWN commits), stored in the fixture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eras", default=",".join(ERAS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for era in args.eras.split(","):
        fixture = era_fixture(args.generated, era, args.source)
        path = args.output / f"{era}.json"
        path.write_text(json.dumps(fixture, indent=1) + "\n")
        print(f"{path}: {', '.join(f'{c} {len(s)} scopes' for c, s in fixture['configs'].items())}; fallback {len(fixture['fallback'])}")


if __name__ == "__main__":
    main()
