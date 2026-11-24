"""Write tests/fixtures/v8_shift_inventory.json.gz, the copy of the CROWN shift inventory that tests/test_v8_shifts.py
compares with the shifts the analysis declares:

    python scripts/make_shift_inventory_fixture.py <shift_inventory_final.json>

The source is the inventory of sm_config with -DSHIFTS=all that the CROWN tests pin (v8 spec 3.2,
docs/analysen/v8/v8_spec_files/shift_inventory_final.json): per era, sample type and scope the shift names by
family. The copy keeps the pairs (names without Up/Down) of the scopes produced with shifts (V3: et, mt, tt, em) and
the kind of every family; it checks that every pair has both directions and that the counts are those of the source.
"""
import argparse
import gzip
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "v8_shift_inventory.json.gz"
SCOPES = ("et", "mt", "tt", "em")  # V3: ee and mm are produced without shifts (tag smrun2_binned_v8_ll)
DIRECTIONS = ("Up", "Down")


def compact(source: dict, name: str) -> dict:
    kinds = {}
    for families in source["families"].values():
        for family, entry in families.items():
            if kinds.setdefault(family, entry["kind"]) != entry["kind"]:
                raise ValueError(f"{family}: {kinds[family]} in one era, {entry['kind']} in another")
    pairs = {}
    for surface, scopes in source["surfaces"].items():
        pairs[surface] = {}
        for scope in SCOPES:
            if scope not in scopes:
                continue
            entry = scopes[scope]
            by_family = {family: sorted(names) for family, names in entry["pairs_by_family"].items()}
            names = [pair for family in by_family.values() for pair in family]
            if len(names) != len(set(names)) or len(names) != entry["n_pairs"]:
                raise ValueError(f"{surface} {scope}: {len(names)} pairs in the families, {entry['n_pairs']} counted")
            if sorted(entry["shifts"]) != sorted(f"{pair}{direction}" for pair in names for direction in DIRECTIONS):
                raise ValueError(f"{surface} {scope}: the shifts are not the Up and Down of the pairs")
            if set(by_family) - set(kinds):
                raise ValueError(f"{surface} {scope}: families without a kind: {sorted(set(by_family) - set(kinds))}")
            pairs[surface][scope] = by_family
    return {
        "description": "Pairs (shift names without Up/Down) of sm_config with -DSHIFTS=all per <era>/<sample type>, scope and family; a shifted column is <column>__<pair>Up/Down.",
        "source": name, "commit": source["commit"], "branch": source["branch"], "frozen_nominal_commit": source["frozen_nominal_commit"],
        "scopes": list(SCOPES), "kinds": kinds, "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="shift_inventory_final.json of the v8 spec")
    parser.add_argument("--output", type=Path, default=FIXTURE)
    args = parser.parse_args()
    fixture = compact(json.loads(args.source.read_text()), args.source.name)
    with open(args.output, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:  # mtime 0: the same bytes for the same content
        handle.write(json.dumps(fixture, indent=None, sort_keys=True, separators=(",", ":")).encode())
    surfaces = fixture["pairs"]
    print(f"{args.output}: {len(surfaces)} surfaces, {sum(len(scopes) for scopes in surfaces.values())} scopes, {len(fixture['kinds'])} families, CROWN {fixture['branch']} {fixture['commit'][:7]}")


if __name__ == "__main__":
    main()
