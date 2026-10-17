#!/usr/bin/env python3
"""Print the sorted branch names of the `ntuple` tree of a CROWN file (local path or root:// URL).

tests/fixtures/branches/sm2018_binned_v2/<channel>.txt are its output for one MC ttbar file per channel of
sm2018_binned_v2, the check of the generated per-era fixtures (scripts/branches_from_crown.py); with the first v7
files it gives the fixtures of real n-tuples.
"""
import sys

import uproot

print("\n".join(sorted(uproot.open(sys.argv[1])["ntuple"].keys())))
