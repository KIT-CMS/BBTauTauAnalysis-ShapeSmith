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

Two run configurations per production. The ShapeSmith core keeps the process table, the estimator
and the control variables per Analysis, so the tau channels (gen-match splits, jet-fake estimate)
and the light-dilepton controls (unsplit MC, no estimate) are separate Analyses. Both read the same
ntuples and share the skim directory (their channels do not overlap); the outputs are separate.

```bash
# tau channels et/mt/tt: nominal shapes and, with control_regions: true, the named tau control regions
shapesmith validate  -c configs/sm2018_binned_v4.yaml
shapesmith skim      -c configs/sm2018_binned_v4.yaml                   # once per production; --samples TT,SingleMuon for a subset
shapesmith hist      -c configs/sm2018_binned_v4.yaml --control --skip-systematics
shapesmith estimate  -c configs/sm2018_binned_v4.yaml --control
shapesmith plot      -c configs/sm2018_binned_v4.yaml --control
# light-dilepton controls em/mm/ee
shapesmith validate  -c configs/sm2018_binned_v4_dilepton.yaml
shapesmith skim      -c configs/sm2018_binned_v4_dilepton.yaml
shapesmith hist      -c configs/sm2018_binned_v4_dilepton.yaml --control --regions all --skip-systematics
shapesmith plot      -c configs/sm2018_binned_v4_dilepton.yaml --control --channels mm,ee --variables yield,m_vis,pt_vis,n_jets
# with NN friends (nn_friend: true, friend under ntuples.friends, re-skim):
shapesmith hist      -c configs/sm2018_binned_v4.yaml && shapesmith estimate -c configs/sm2018_binned_v4.yaml
shapesmith sync      -c configs/sm2018_binned_v4.yaml && shapesmith datacards -c configs/sm2018_binned_v4.yaml && shapesmith fit -c configs/sm2018_binned_v4.yaml
shapesmith ml-export -c configs/sm2018_binned_v4.yaml
```

Switches of the tau run YAML: `sample_list` (which `inventory/<sample_list>.txt`), `jet_fakes: mc|ff`,
`embedding`, `nn_friend`, `control_regions`. The dilepton run YAML takes `sample_list` only.
`sample_list` selects the sample inventory independently of the ntuple production in `ntuples.base`:
the v4 ntuples use `sample_list: sm2018_binned_v2` because the sample nicks are unchanged.

The control regions of both Analyses and how to read them: [docs/control_regions.md](docs/control_regions.md).

## Where things are defined

| What | Where |
|---|---|
| produced samples (nick + DBS path per production), groups, data-stream routing, normalisation | `inventory/*.txt`, `samples.py` |
| era, luminosity, channel sets, working points, b-tag bins, friend columns, class names | `constants.py` |
| tau channels: baseline/skim selection, estimation and control regions, MC weights, trigger chains, gen-match splits | `selection.py` |
| dilepton channels: baseline/skim selection, b-tag bins, same-sign region, MC weights | `dilepton_selection.py` |
| process tables (tau: gen-match split, dilepton: unsplit), NN categories, control variables and binning | `processes.py` |
| b-tag weight variations, lnN table, colours, labels, axis titles of all six channels | `systematics.py` |
| tau analysis assembly, switches, ML export | `analysis.py` |
| dilepton analysis assembly | `dilepton_analysis.py` |

Sample nicks changed in the sample database in 2026-08 (campaign suffix); `sm2018_binned_v1`
carries the old, `sm2018_binned_v2` the new names. The inventories list the DBS path of every
sample, so either production resolves against the current database.

`tests/fixtures/branches_<channel>.txt` list the branches of one ttbar ntuple per channel of the
`sm2018_binned_v4` production (`scripts/dump_branches.py <root:// URL>`); the tests check every
selection, weight and variable against them.

## Prerequisites outside this repository

NN friend trees (`predicted_class`, `predicted_max_value`) from a trained model + CROWN `sm_ml.py`;
fake-factor friends (`fake_factor`, `fake_factor_1/2`) from TauFakeFactors + CROWN; embedding ntuples.
Until they exist: `nn_friend: false`, `jet_fakes: mc`, `embedding: false`.
