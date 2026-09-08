#!/usr/bin/env python3
"""Print the sorted branch names of the `ntuple` tree of a CROWN file (local path or root:// URL).

Regenerates inventory/2018_v15_<channel>.txt, the fixtures the tests compare the configuration against.
"""
import sys

import uproot

print("\n".join(sorted(uproot.open(sys.argv[1])["ntuple"].keys())))
