# BBTauTauAnalysis-ShapeSmith

Non-resonant HH→bbττ (2018 UL, NanoAODv15) defined for
[ShapeSmith](https://github.com/KIT-CMS/ShapeSmith). Design and physics choices:
`ShapeSmith/docs/superpowers/specs/2026-09-05-shapesmith-core-and-bbtautau-design.md`.

Expected layout: this repository, the ShapeSmith core and the KingMaker sample database as
sibling checkouts (`../ShapeSmith`, `../KingMaker_sample_database`), no submodules.

## Setup

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
source ~/.venvs/shapesmith/bin/activate          # venv with `pip install -e ../ShapeSmith`
pip install -e ".[test]" && pytest
voms-proxy-init --voms cms --valid 192:00        # dCache access for skim
```

## Running

```bash
shapesmith validate  -c configs/sm2018_binned_v2.yaml
shapesmith skim      -c configs/sm2018_binned_v2.yaml                   # once per production; --samples TT,SingleMuon for a subset
shapesmith hist      -c configs/sm2018_binned_v2.yaml --control --skip-systematics
shapesmith estimate  -c configs/sm2018_binned_v2.yaml --control
shapesmith plot      -c configs/sm2018_binned_v2.yaml --control
# with NN friends (nn_friend: true, friend under ntuples.friends, re-skim):
shapesmith hist      -c configs/sm2018_binned_v2.yaml && shapesmith estimate -c configs/sm2018_binned_v2.yaml
shapesmith sync      -c configs/sm2018_binned_v2.yaml && shapesmith datacards -c configs/sm2018_binned_v2.yaml && shapesmith fit -c configs/sm2018_binned_v2.yaml
shapesmith ml-export -c configs/sm2018_binned_v2.yaml
```

Switches in the run YAML: `sample_list` (which `inventory/<sample_list>.txt`), `jet_fakes: mc|ff`,
`embedding`, `nn_friend`.

`sample_list` selects the sample inventory independently of the ntuple production path in
`ntuples.base`. For example, v3 ntuples can use `sample_list: sm2018_binned_v2` when their
sample nicks are unchanged. The former switch `production` has been renamed to `sample_list`;
update existing run YAMLs and CLI overrides (`--set switches.sample_list=...`).

## Where things are defined

| What | Where |
|---|---|
| produced samples (nick + DBS path per production), groups, normalisation | `inventory/*.txt`, `samples.py` |
| baseline/skim selection, estimation regions, MC weights, trigger chains, gen-match splits | `selection.py` |
| processes per switch, NN categories, control variables and binning | `processes.py` |
| b-tag weight variations, lnN table, colours, labels, axis titles | `systematics.py` |
| era, luminosity, working points, friend columns, class names | `constants.py` |
| analysis assembly, switches, ML export | `analysis.py` |

Sample nicks changed in the sample database in 2026-08 (campaign suffix); `sm2018_binned_v1`
carries the old, `sm2018_binned_v2` the new names. The inventories list the DBS path of every
sample, so either production resolves against the current database.

## Prerequisites outside this repository

NN friend trees (`predicted_class`, `predicted_max_value`) from a trained model + CROWN `sm_ml.py`;
fake-factor friends (`fake_factor`, `fake_factor_1/2`) from TauFakeFactors + CROWN; embedding ntuples.
Until they exist: `nn_friend: false`, `jet_fakes: mc`, `embedding: false`.
