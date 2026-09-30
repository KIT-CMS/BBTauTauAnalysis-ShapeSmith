# BBTauTauAnalysis-ShapeSmith

Non-resonant HH→bbττ (2018 UL, NanoAODv15) defined for
[ShapeSmith](https://github.com/KIT-CMS/ShapeSmith). Design and physics choices:
`ShapeSmith/docs/superpowers/specs/2026-09-05-shapesmith-core-and-bbtautau-design.md`; embedding, fake factors and
sample lists: the specs of 2026-09-29 with their amendments of 2026-09-30.

Expected layout: this repository, the ShapeSmith core and the KingMaker sample database as
sibling checkouts (`../ShapeSmith`, `../KingMaker_sample_database`), no submodules.

## Setup

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh
source ~/.venvs/shapesmith/bin/activate          # venv with `pip install -e ../ShapeSmith`
pip install -e ".[test]" && pytest
voms-proxy-init --voms cms --valid 192:00        # dCache access for skim
```

## GitHub Actions

The test workflow checks out `KIT-CMS/ShapeSmith` at `main` beside this repository
and installs that checkout before installing the analysis, so a core change is pushed first
(or the analysis workflow is re-run after the core push).

## Running

Two run configurations per production. The tau channels (gen-match splits, jet-fake estimate)
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
shapesmith inspect   output/sm2018_binned_v4/shapes.root --unchanged   # variations equal to their nominal
```

Switches of the tau run YAML (typed in `switches.py`; an unknown or mistyped switch fails):

| Switch | Values | Effect |
|---|---|---|
| `sample_lists` | inventory names (required) | the samples to process, independent of the ntuple production in `ntuples.base` |
| `jet_fakes` | `mc` (default), `ff` | MC jet fakes + ABCD QCD, or `jetFakes` from the fake-factor friend (data minus genuine and lepton fakes in `anti_iso`) |
| `embedding` | `false` (default), `true` | genuine ττ from the embedded samples (`EMB`) instead of the T parts of the MC; needs the embedding inventory in `sample_lists` |
| `nn_friend` | `false` (default), `true` | NN categories from the NN friend |
| `control_regions` | `false` (default), `true` | wider skim plus the named pass/fail control regions; needs `jet_fakes: mc` |
| `shape_systematics` | `true` (default), `false` | the CROWN shifts of the embedded samples and of the fake-factor friend as shape variations |

The dilepton run YAML takes `sample_lists` only. Override a switch on the command line with
`-s switches.jet_fakes=ff`. `--skip-systematics` skips the b-tag weight variations and the CROWN shifts.

The control regions of both Analyses and how to read them: [docs/control_regions.md](docs/control_regions.md).

## Sample lists

`inventory/<name>.txt` is an unchanged copy of a KingMaker sample list a production ran with: one nick per line,
no comments, no blank lines. ShapeSmith looks up every nick in the configured `sample_database` (kind, cross
section, event count, generator weight); a missing nick fails with the list of all missing nicks (the database
may have renamed them; use the database checkout of the production). A changed KingMaker list gets a new
inventory name and the old inventory is never edited; productions that ran with the same list share one
inventory (v4 uses `sm2018_binned_v2`). A run YAML names every list of its production, e.g.
`sample_lists: [sm2018_binned_v3, sm2018_embedding]`; a nick in two lists fails.

| Inventory | Source (CROWN `analysis_configurations/bbtautau`) | Used by |
|---|---|---|
| `sm2018_binned_v2.txt` | `sample_list/sm_2018_binned.txt` @ `f99a322` "chore: adapt to sample database update" | productions sm2018_binned_v2, v4 |
| `sm2018_binned_v3.txt` | `sample_list/sm_2018_binned.txt` @ `85c437a` "fix: added DYJetsToLL_M-50 for events with zero LHE partons" | the next production |
| `sm2018_embedding.txt` | `sample_list/sm_2018_embedding.txt` @ `df97b0b` "feat: add the 2018 embedding sample list" | the embedding production (et, mt, tt) |

Check a copy with `diff <(git -C <crown>/analysis_configurations/bbtautau show <commit>:sample_list/<file>) inventory/<name>.txt`
(the subject finds the commit again after a rebase).

`sm2018_binned_v3` adds the inclusive DY M-50 amcatnloFXFX sample, of which only the `npartons == 0`
events are kept (`SAMPLE_CUTS` in `samples.py`): the LHEFilterPtZ bins lack every zero-parton event
and cover `npartons >= 1`. It needs a production that contains this sample; the v4 run YAMLs keep
`sm2018_binned_v2`. Only this sample reads `npartons`, so the skims of the other samples stay
valid when an existing skim directory switches to `sm2018_binned_v3`.

Routing (`samples.route`): data streams go to the channels whose trigger they carry; embedded samples
(`EmbeddingRun…`) by their final state, `_mutau_` → mt, `_eltau_` → et, `_tautau_` → tt (group `EMB`) and
`_muemb_` → mm (group `MUEMB`, for the tau-ID measurement); every other sample by its nick prefix. A channel
holds only the samples of its processes. The skim contract records `nevents` for every kind: fix the mutau 2018D
event count in the sample database (21,362,177 → 25,419,943) before the first embedding skim, or re-skim that
sample once with `shapesmith skim --force --samples EmbeddingRun2018D_cwinter-embedding_2018UL_mutau`.

`tests/fixtures/datasets.json` holds the unchanged database entries (`KingMaker_sample_database` `94bd5a0f`) of
exactly the nicks in `inventory/`; a new inventory adds its entries in the same change.

## Embedding and fake factors

- **Processes.** Gen-match splits T (genuine ττ, the lepton leg from a τ decay), L (lepton fakes), J (jet → τh) for
  DY, TT, ST, VV, TTV and EWK. With `embedding: true`, `EMB` (genmatch T) replaces every T part; `TTT` stays as
  an auxiliary process (booked in nominal, in no datacard or plot) for the ttbar contamination
  `CMS_htt_emb_ttbar_2018` (EMB ± 10 % of TTT). With `jet_fakes: ff`, `jetFakes` replaces the J parts and W.
- **EMB weights** (Part-A contract): `emb_genweight`, the embedding selection SFs, the embedding lepton ID/iso SFs
  (mt `id_wgt_mu_1`, `iso_wgt_mu_1`; et `id_wgt_ele_1`, `iso_wgt_ele_1`), the vsJet SF of genuine τh, vsE/vsMu and the
  trigger SF; no pileup and no b-tag weight. They carry their own names (`emb_…`), so regions that replace MC
  weights (the tau-fail regions, the mt muon anti-isolation region) leave them unchanged. In the anti-isolated
  regions genuine taus keep the Medium pass SF, as in TauFakeFactors and the legacy analysis.
- **Shifts** (`systematics.py`, only with `shape_systematics: true`). The analysis declares every CROWN shift it
  reads. For a sample kind with declared shifts the skim fails on a declared shift without a shifted branch and on an
  undeclared `c__X` branch of a needed column (main n-tuple or friend):
  - embedding (only with `embedding: true`): `CMS_scale_t_emb_dm{0,1,1011}_<wp>_2018` (branches `__embTauEs{1prong0pizero,1prong1pizero,3prong}`)
    and `CMS_eff_t_emb_dm{0,1,1011}_pt{20to40,40toInf}_<wp>_2018` (`__embVsJetTauDM{0,1,1011}Pt{20to40,40toInf}`),
    `wp` = `vsEleTight` in et, `vsEleVVLoose` in mt and tt; tt has no pt20to40 shifts. 9 pairs in et and mt, 6 in tt.
  - fake factors (only with `jet_fakes: ff`): `CMS_ff_<key>_<channel>_2018` with branch suffix `__<key>`, for the
    17 keys of `FF_SHIFTS_LT` (et, mt) and the 34 of `FF_SHIFTS_TT` (tt, both legs), on data, MC and embedding.
    They follow the SM 2018 payload; a new payload with other keys fails the skim until the lists follow it.
  - `jetFakes` is built for every shift found on data or a subtracted process in `anti_iso`, so the embedding shifts
    reach it too.
- **One skim for all switches.** A skim made with `embedding: true, jet_fakes: ff` (FF friend configured) serves all
  four combinations. Skims made without the friend or the shifts are incompatible with it: the first such run needs
  `shapesmith skim --force --samples …` for the samples it lists.

## Fake-factor measurement

`configs/ff_sm2018_binned_v4.yaml` runs `bbtautau_shapesmith.ff_measurement:build`, the SM 2018 fake-factor
measurement in jvoss's TauFakeFactors method (QCD and ttbar fake factors, fractions, the QCD DR->SR correction and the
non-closures), with its own skim (both charges, the lepton vetoes stored for the ttbar scale regions):

```bash
shapesmith skim    -c configs/ff_sm2018_binned_v4.yaml
shapesmith measure -c configs/ff_sm2018_binned_v4.yaml [--suggest-binning]   # -> output/.../fake_factors/2018/
```

The regions come from `cuts.py`; every difference to jvoss's configuration is listed in the module docstring. The
binning, fits and bandwidths are `ff_tables.py`, generated once by `scripts/make_ff_tables.py` from TauFakeFactors'
resolved configuration (do not edit by hand). `scripts/ff_parity.py` is the algorithm parity with TauFakeFactors: it
runs the measurement on jvoss's preselection files with his regions and weights and compares every payload value,
edge and fitted curve with his payload and his pickled fits, and `--suggest-binning` with his `adjust_binning.py`.
A new payload goes into a new dated directory of the CROWN analysis (`payloads/fake_factors/sm/fake-factors-<date>/`)
and needs a fresh friend tag; `constants.FF_SHIFTS_LT/TT` must follow its keys.

## Tau-ID and ES measurement of the embedded taus

`tau_id_measurement.py` (`configs/tau_id_es_2018.yaml`) is another measurement Analysis: the successor of smhtt_ul
`tauID_SFs_dev`, run by the core measurement `shapesmith.measurements.tau_id_es`. It needs the production of CROWN
`sm_tau_id_measurement_config` (list `sm2018_tau_id_measurement`, scopes mt and mm, all shifts), which does not exist
yet.

- **Selection** as smhtt_ul `config/shapes` (special `TauID_ES`, 2018) with bbtautau column names: mt tag and probe
  with `mt_1 < 65`, `IsoMu24 || IsoMu27` and pT > 25 / 20 GeV, the nine categories DM0, DM1, DM1011 (pT >= 20 GeV) and
  their [20, 40) and [40, 200] GeV bins; the mm control region 70-110 GeV in one bin. m_vis bins per working-point
  combination and category from the predecessor's shapes (`tau_id_binning.py`).
- **Processes.** mt: EMB (genmatch 4/5; no tau ID, vsEle, vsMu or ES correction), ZL, ZJ, TTL, TTJ, STL, STJ, VVL, VVJ,
  W and QCD = same-sign data minus all of them (negative bins clipped); TTT is the auxiliary template of
  `CMS_emb_ttbar_contamination_Run2018`. mm: MUEMB (genmatch 2/2, muon SFs of both legs), W, TTL, VVL and QCD =
  same-sign data minus MUEMB and W. MC and embedding weights are the predecessor's: KIT muon ID/iso SFs, the trigger
  weight `mu24` for 25 <= pT < 28 GeV and `mu27` above 28 GeV.
- **Switches** `vsjet_wp` (Medium, Tight) and `vsele_wp` (VVLoose, Tight). The skim takes the loosest of them, and the
  four combinations are regions `wp_<vsjet>_<vsele>`, so one skim serves all four runs.
- **ES grid**: `es-200` ... `es+200` (0.2 % steps, units of 0.1 %), template column variations of the embedded sample
  in the nominal region, derived from the nominal columns: the tau four-momentum scaled (pt_2, mass_2, m_vis from the
  four-vectors), the MET corrected by (1 - s) times the tau pT, mt_1 from that MET.
- **MC shape uncertainties** (`tau_id_systematics.py`): the predecessor's set with MorphingTauID2017's names, from the
  CROWN shifts `<quantity>__<shift>` of `sm_tau_id_measurement_config` and the jet-fake and top-pT weights; mt only,
  nominal region only. The vsJet SF families `CMS_eff_t_dm*` stay no-ops as in the predecessor; the production's POG
  SF components are read as their own variations, unused by the datacards until their treatment is decided.
- **CMSSW**: MorphingTauID2017 exists only with jvoss's local changes, versioned in `patches/` (see
  `patches/README.md`).

## Where things are defined

| What | Where |
|---|---|
| produced samples (verbatim KingMaker sample lists), groups, routing, per-sample cuts, normalisation | `inventory/*.txt`, `samples.py` |
| the switches and their checks | `switches.py` |
| era, luminosity, channel sets, working points, b-tag bins, friend columns, class names, FF shift keys, embedding shift tables | `constants.py` |
| tau channels: baseline/skim selection, gen-match splits, estimation and control regions | `cuts.py` |
| MC, embedding, fake-factor and dilepton weights, trigger chains | `weights.py` |
| process tables (tau: gen-match split, dilepton: unsplit) | `processes.py` |
| NN categories, control variables and binning | `variables.py` |
| b-tag weight variations, embedding and fake-factor shifts, lnN table | `systematics.py` |
| colours, labels, axis titles of all six channels | `style.py` |
| tau analysis assembly, estimators, ML export | `analysis.py` |
| fake-factor measurement: regions, processes, legs; its tables | `ff_measurement.py`, `ff_tables.py` |
| dilepton channels and analysis assembly | `dilepton.py` |
| tau-ID/ES measurement: selection, processes, ES grid, assembly; its m_vis bins; its MC shape uncertainties | `tau_id_measurement.py`, `tau_id_binning.py`, `tau_id_systematics.py` |

`tests/fixtures/branches_<channel>.txt` list the branches of one ttbar ntuple per channel of the
`sm2018_binned_v4` production (`scripts/dump_branches.py <root:// URL>`); the tests check every
selection, weight and variable against them, and the embedding columns against them plus the Part-A output
contract until a branch fixture of a real embedding file exists. `tests/fixtures/skim_contracts_v4.json` holds the
stored contracts and Parquet columns of one data and one MC skim per channel of v4 (from the golden references of
2026-09-30); `tests/test_contracts_v4.py` checks that the v4 run YAMLs still reuse them.

## Prerequisites outside this repository

- NN friend trees (`predicted_class`, `predicted_max_value`) from a trained model + CROWN `sm_ml.py`. With embedding
  shifts the NN friend must be produced with `--shifts all`, or the NN category and score stay nominal under them.
- Fake-factor friends (`fake_factor`, `fake_factor_1/2` and their shifts) from CROWN `fake_factors_friend_config.py`,
  on data, MC and embedding. A new FF payload gets a new friend tag (a new `ntuples.friends` base); the skim contract
  records the friend bases, so the stored skims then need `skim --force`.
- Embedding ntuples under the same `ntuples.base` and production tag as the data/MC they are combined with, produced
  with `--scopes et,mt,tt --shifts all`.

Until they exist: `nn_friend: false`, `jet_fakes: mc`, `embedding: false`.
