# BBTauTauAnalysis-ShapeSmith

Non-resonant HH→bbττ (Run 2 UL: 2016preVFP, 2016postVFP, 2017, 2018; NanoAODv15) defined for
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

The tau channels (gen-match splits, jet-fake estimate) and the light-dilepton controls (unsplit MC, no estimate) are
separate Analyses on the same ntuples, with separate outputs. From v6 on every run has its own skim directory: the
tau control run, the fake-factor run (with em as the ttbar control channel), the dilepton run and the fake-factor
measurement (its two runs share one).

```bash
# tau channels et/mt/tt, control run: MC jet fakes + ABCD QCD, embedding in et/mt, QCD in the control regions, DY correction
shapesmith validate  -c configs/sm2018_binned_v6.yaml
shapesmith skim      -c configs/sm2018_binned_v6.yaml                   # once per production; --samples TT,SingleMuon for a subset
shapesmith hist      -c configs/sm2018_binned_v6.yaml --control --skip-systematics   # nominal + the regions of the estimates
shapesmith estimate  -c configs/sm2018_binned_v6.yaml --control
shapesmith plot      -c configs/sm2018_binned_v6.yaml --control           # nominal; once more per --region btag0_os_pass (btag1, btag2, btag3p)
# light-dilepton controls em/mm/ee; the DY table from mm and ee (dy_correction: none there)
shapesmith skim      -c configs/sm2018_binned_v6_dilepton.yaml
shapesmith hist      -c configs/sm2018_binned_v6_dilepton.yaml --control --regions all --skip-systematics
shapesmith plot      -c configs/sm2018_binned_v6_dilepton.yaml --control --region btag1
python scripts/derive_dy_correction.py -c configs/sm2018_binned_v6_dilepton.yaml --ttbar-scale em   # -> DY_NJETS_BTAG
# fake-factor measurement: the mixed payload of the friend ff_v2 and the cross-check without embedding
shapesmith skim      -c configs/ff_sm2018_binned_v6.yaml
shapesmith measure   -c configs/ff_sm2018_binned_v6.yaml && shapesmith measure -c configs/ff_sm2018_binned_v6_noemb.yaml
# fake-factor run (friend ff_v2) with em as the ttbar control channel: control plots, then the background-only fit
shapesmith skim      -c configs/sm2018_binned_v6_ff.yaml
shapesmith hist      -c configs/sm2018_binned_v6_ff.yaml --control --skip-systematics && shapesmith estimate -c configs/sm2018_binned_v6_ff.yaml --control
shapesmith hist      -c configs/sm2018_binned_v6_ff.yaml --skip-systematics && shapesmith estimate -c configs/sm2018_binned_v6_ff.yaml   # categories nominal, em_tt
shapesmith datacards -c configs/sm2018_binned_v6_ff.yaml && shapesmith fit -c configs/sm2018_binned_v6_ff.yaml --background-only   # -> datacards/background_fit.md
# web galleries (after plot or measure; target, column and texts from the web: block of each run YAML), in this
# order: the first publication fixes the order of the columns and channels
shapesmith publish   -c configs/sm2018_binned_v6.yaml -s switches.embedding=false -s output_dir=../output/sm2018_binned_v6_noemb \
                     -s web.variant=classic -s web.label=classic -s web.description=null   # ~sdaigler/sm2018_binned_v6 (+ title, about), column classic: et, mt, tt
shapesmith publish   -c configs/sm2018_binned_v6_dilepton.yaml           # column classic: em, mm, ee
shapesmith publish   -c configs/sm2018_binned_v6.yaml --channels et,mt   # column emb
shapesmith publish   -c configs/sm2018_binned_v6_ff.yaml --channels et,mt,em   # column emb ff
shapesmith publish   -c configs/sm2018_binned_v6_ff.yaml --channels tt --variant ff --label ff \
                     --description "MC genuine ττ + jet→τh fakes from the fully corrected fake factor (friend ff_v2); tt only, it has no embedding."   # column ff: tt
shapesmith publish   -c configs/ff_sm2018_binned_v6.yaml                 # ~sdaigler/ff_sm2018_binned_v6: the FF measurement
shapesmith publish   -c configs/ff_sm2018_binned_v6_noemb.yaml           # its cross-check column
# with NN friends (nn_friend: true, fit_categories: nn, friend under ntuples.friends, re-skim):
shapesmith hist      -c configs/sm2018_binned_v6_ff.yaml && shapesmith estimate -c configs/sm2018_binned_v6_ff.yaml
shapesmith sync      -c configs/sm2018_binned_v6_ff.yaml && shapesmith datacards -c configs/sm2018_binned_v6_ff.yaml && shapesmith fit -c configs/sm2018_binned_v6_ff.yaml
shapesmith ml-export -c configs/sm2018_binned_v6_ff.yaml
shapesmith inspect   output/sm2018_binned_v6_ff/shapes.root --unchanged   # variations equal to their nominal
```

`publish` copies the plots of a run (for the FF measurement those in `fake_factors/2018/`) into a static web gallery
under `/web/sdaigler/public_html/`; run it after `plot` or `measure` with the same `-s` overrides. The `web:` block of
each run YAML sets the gallery directory, the column (`variant`), its label and description; the title and the About
text of the control gallery are in `sm2018_binned_v6.yaml`. Publishing a column again replaces it and copies only
changed plots; the first publish of a column fixes its position. The classic column is the run without embedding:
`hist`, `estimate` and `plot` of `sm2018_binned_v6.yaml` with `-s switches.embedding=false
-s output_dir=../output/sm2018_binned_v6_noemb`; `web.description=null` keeps the description of the dilepton YAML.
tt has no embedding, so it is only in the columns classic and ff (the tt of the fake-factor run); emb and emb ff hold
et and mt, emb ff also the control channel em. `publish` takes every plotted region: classic and emb have the
diagnostic b-tag regions `btag0_os_pass` to `btag3p_os_pass` (the fake-factor run has the nominal region only).

**v7 (Run 2).** One run YAML per run type serves every era: `-s era=<2016preVFP | 2016postVFP | 2017 | 2018>` fills
`{era}` in the sample lists, paths and web texts (ShapeSmith core). Productions: P1 `smrun2_tau_id_v1` (tau-ID
measurement and tt trigger), P3 `smrun2_binned_v7` with the friends `fastmtt`, `ff_v7` and `ff_v7_w`; skims, outputs
and ML folds under `/ceph/sdaigler/shapesmith/<tag>/{skims,output,ml}/<run>/<era>`; galleries per era
`~sdaigler/smrun2_binned_v7/<era>` (columns classic, emb, emb ff, emb ff with W) and `~sdaigler/ff_smrun2_binned_v7/<era>`.

| Run YAML | Run type | Notes |
|---|---|---|
| `smrun2_binned_v7.yaml` | tau control, column emb | MC jet fakes, ABCD QCD, embedding in et, mt, tt, control regions |
| `smrun2_binned_v7_noemb.yaml` | tau control without embedding, column classic | the skims of `smrun2_binned_v7.yaml` |
| `smrun2_binned_v7_dilepton.yaml` | em, mm, ee, column classic | source of the DY table of the era |
| `ff_smrun2_binned_v7.yaml`, `ff_smrun2_binned_v7_w.yaml` | FF measurement, QCD + ttbar and QCD + W+jets + ttbar | one skim; `-s era=2016` measures both halves together |
| `smrun2_binned_v7_ff.yaml`, `smrun2_binned_v7_ff_w.yaml` | FF run (friend ff_v7 / ff_v7_w) with em, datacards, columns emb ff / emb ff with W | `shapesmith combine ... --era` per era |
| `smrun2_tau_id_v1_id_es.yaml` | tau-ID/ES measurement of the embedded taus | four WP runs, then `--merge` |
| `smrun2_tau_id_v1_trigger.yaml` | tau-leg trigger efficiency of the tt embedding (E33) | needs the tau-ID/ES payload of the era |

```bash
E=2017; S="-s era=$E"                              # the same for 2016preVFP, 2016postVFP, 2018
# P1: tau-ID/ES (spec 5.4), then the tt-trigger efficiency (E33)
shapesmith skim    -c configs/smrun2_tau_id_v1_id_es.yaml $S
for jet in Medium Tight; do for ele in VVLoose Tight; do
  shapesmith measure -c configs/smrun2_tau_id_v1_id_es.yaml $S -s switches.vsjet_wp=$jet -s switches.vsele_wp=$ele; done; done
shapesmith measure -c configs/smrun2_tau_id_v1_id_es.yaml $S --merge
shapesmith skim    -c configs/smrun2_tau_id_v1_trigger.yaml $S
for cls in Medium VVVLoose_not_Medium; do for variant in nominal signal_mc tag_pt mt30 monitor; do   # no monitor in 2018
  shapesmith measure -c configs/smrun2_tau_id_v1_trigger.yaml $S -s switches.wp_class=$cls -s switches.variant=$variant; done; done
shapesmith measure -c configs/smrun2_tau_id_v1_trigger.yaml $S --merge
# P3: dilepton controls and the DY table of the era (constants.DY_NJETS_BTAG)
D=configs/smrun2_binned_v7_dilepton.yaml
shapesmith skim -c $D $S && shapesmith hist -c $D $S --control --regions all --skip-systematics && shapesmith plot -c $D $S --control
python scripts/derive_dy_correction.py -c $D $S --ttbar-scale em
# tau control runs, columns emb and classic (one skim)
T=configs/smrun2_binned_v7.yaml; N=configs/smrun2_binned_v7_noemb.yaml
shapesmith skim -c $T $S
for C in $T $N; do shapesmith hist -c $C $S --control --regions all --skip-systematics && shapesmith estimate -c $C $S --control && shapesmith plot -c $C $S --control; done
# fake factors, both variants (2016: -s era=2016, both halves; the halves alone are the cross-check)
shapesmith skim    -c configs/ff_smrun2_binned_v7_w.yaml $S
shapesmith measure -c configs/ff_smrun2_binned_v7_w.yaml $S --suggest-binning   # edges -> ff_tables.PER_ERA_EDGES
shapesmith measure -c configs/ff_smrun2_binned_v7.yaml $S && shapesmith measure -c configs/ff_smrun2_binned_v7_w.yaml $S
# FF runs (after the friends ff_v7, ff_v7_w), datacards, background-only fit, then the Run-2 combination
for F in configs/smrun2_binned_v7_ff.yaml configs/smrun2_binned_v7_ff_w.yaml; do
  shapesmith skim -c $F $S
  shapesmith hist -c $F $S --control --skip-systematics && shapesmith estimate -c $F $S --control && shapesmith plot -c $F $S --control
  shapesmith hist -c $F $S --skip-systematics && shapesmith estimate -c $F $S && shapesmith datacards -c $F $S && shapesmith fit -c $F $S --background-only
done
shapesmith combine -c configs/smrun2_binned_v7_ff.yaml --era 2016preVFP --era 2016postVFP --era 2017 --era 2018 \
                   -o /ceph/sdaigler/shapesmith/smrun2_binned_v7/combination/ff --background-only --validate
# galleries of the era, in this order (the first publication fixes the order of the columns and channels)
for C in $N $D $T configs/smrun2_binned_v7_ff.yaml configs/smrun2_binned_v7_ff_w.yaml configs/ff_smrun2_binned_v7.yaml configs/ff_smrun2_binned_v7_w.yaml; do
  shapesmith publish -c $C $S; done
```

The FF runs keep the friend shifts as shape variations (`shape_systematics: true`, E34: the FF variations stay in the
nominal cards). They are CROWN shifts of the friend (`friend_common.py`), so P5 (`ff_v7`, `ff_v7_w`) must be produced
with `--shifts all` as `ff_v2` in v6, not with the `--shifts None` that spec E34 names for P2-P5: the skim refuses a
declared FF shift without its shifted branch (`-s switches.shape_systematics=false` runs without them).

Every era and 2016 half has its DY table (`constants.DY_NJETS_BTAG`, from its `smrun2_binned_v7_dilepton.yaml` run);
the 2016 FF measurement takes the luminosity-weighted mean of the tables of both halves. `--workers` stays at most 24
on portal2; the dilepton `hist` needs up to ~45 GB per worker for the DY and data samples of 2017 and 2018.

The v5 run YAMLs (`*v5*.yaml`, galleries `sm2018_binned_v5` and `ff_sm2018_binned_v5`, run as above) share one skim
directory between the tau and the dilepton run. With the v6 code, `hist`, `estimate` and `plot` of tt, em, mm and ee
run on their stored skims with `--skip-systematics` (the v5 embedding shifts are pT-binned, see below); the et and mt
skims need `skim --force` (the tau |η| < 2.3 cut, the electron reconstruction SF of the et MC).

Switches of the tau run YAML (typed in `switches.py`; an unknown or mistyped switch fails):

| Switch | Values | Effect |
|---|---|---|
| `sample_lists` | inventory names (required) | the samples to process, independent of the ntuple production in `ntuples.base` |
| `jet_fakes` | `mc` (default), `ff` | MC jet fakes + ABCD QCD, or `jetFakes` from the fake-factor friend (data minus genuine and lepton fakes in `anti_iso`) |
| `ff_type` | `corrected` (default), `raw` | the FF friend weight of `jetFakes`: corrected, or raw (no DR->SR and non-closure corrections; needs `jet_fakes: ff` and `shape_systematics: false`) |
| `embedding` | `false` (default), `true`, a list of channels (`[et, mt]`) | genuine ττ from the embedded samples (`EMB`) instead of the T parts of the MC, in every tau channel or in the listed ones (each a channel of the run); needs the embedding inventory in `sample_lists` |
| `nn_friend` | `false` (default), `true` | NN categories from the NN friend |
| `fit_categories` | `nn` (default), `control` | the categories of `hist`, `datacards` and `fit`: the NN classes (with `nn_friend`), or one category `nominal` per tau channel with the nominal region in one bin (`yield`), e.g. for `fit --background-only` without NN friends |
| `control_regions` | `false` (default), `true` | wider skim plus the named pass/fail control regions; needs `jet_fakes: mc` |
| `diagnostics` | `false` (default), `true` | QCD in the control regions (et/mt: same-sign data minus MC, tt: ABCD with the fail regions) and the variables `met_uncorrected`, `metphi_uncorrected`, `npv_good`, `rho` (the last two from production v6 on); needs `control_regions: true` |
| `shape_systematics` | `true` (default), `false` | the CROWN shifts of the embedded samples and of the fake-factor friend as shape variations |
| `crown_shifts` | `true` (default), `false` | `false` for a production without CROWN shifts (v7 is nominal only, E34): no embedding tau shifts; the fake-factor friend shifts and the weight variations of the nominal tree (b tag, prefiring) stay |
| `dy_correction` | `none` (default), `njets_btag` | every DY process (ZTT, ZL, ZJ) times the DY data/MC table `DY_NJETS_BTAG` (`constants.py`) of its n_bjets (0, 1, ≥ 2) and n_jets (≤ 2, 3, ≥ 4) |

**em as the ttbar control channel.** A tau run YAML can list `em` in `channels` (e.g. `[et, mt, tt, em]`). It is a
control channel (`Analysis.control_channels`): `datacards` and `fit` put it into every final state, never into one of
its own. It is the dilepton em selection (OS, isolated e and μ, ≥ 2 jets) with the unsplit MC of the dilepton
analysis (ttbar with the top-pT weight), one category `em_tt` with the b tag of the tau baseline counted in one bin,
no further regions, the b-tag and prefiring variations of the tau channels and the lnN `eff_e`, `eff_m` and
`CMS_eff_m_trigger_<era>` (2 %, the muon trigger of mt); `htt_tjXsec` correlates its ttbar with that of the tau channels. The friends of the
tau channels name them (`channels: [et, mt, tt]`), so the em skim looks for none; a tau run with em has its own
`skim_dir` (the em skim of the dilepton run has other columns). The ML labels come from the tau channels only, and
`ml-export` skips the control channel em.

The dilepton run YAML takes `sample_lists`, `dy_correction` (on `DY`) and `diagnostics` (the jet-multiplicity regions
`nj2`, `nj3`, `nj4p` with 0 and ≥ 1 b tags, the diagnostic variables, and the auxiliary flavour templates `DY_b`,
`DY_c`, `DY_l` in mm/ee and `TT_b`, `TT_c`, `TT_l` in em from the GenJet hadron flavour of the leading-score b-pair
jet, booked in the nominal region; data reads no flavour column). The table is the mm and ee data/MC of v6;
`scripts/derive_dy_correction.py -c <dilepton run YAML> [--channels mm,ee] [--ttbar-scale em]` derives it again from
the control shapes of a dilepton run without the correction (`hist --control --regions all`) and prints it in the
format of `constants.py`; `--ttbar-scale em` scales the subtracted ttbar with the em data/MC of the same b-tag region. Override a switch on the command line with
`-s switches.jet_fakes=ff`. `--skip-systematics` skips the b-tag weight variations and the CROWN shifts.

Every run YAML sets `log_level: DEBUG` (every file, booking and fit); `-s log_level=INFO` gives the short log of one run. Each command
writes its log to `<output_dir>/logs/<command>_<time>.log` as well (see the core README, "Logging"). The analysis logs
its switches with their defaults, the sample lists and the skim-time sample cuts at INFO, every sample at DEBUG.

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
| `sm2018_binned_v3.txt` | `sample_list/sm_2018_binned.txt` @ `85c437a` "fix: added DYJetsToLL_M-50 for events with zero LHE partons" | production sm2018_binned_v5 |
| `sm2018_binned_v6.txt` | `sample_list/sm_2018_binned.txt` @ `f9c14c6` "fix: drop the ttZ and ZZ double counting from the SM sample lists" | production sm2018_binned_v6 |
| `sm2018_embedding.txt` | `sample_list/sm_2018_embedding.txt` @ `df97b0b` "feat: add the 2018 embedding sample list" | productions sm2018_binned_v5, v6 (et, mt, tt) |

Check a copy with `diff <(git -C <crown>/analysis_configurations/bbtautau show <commit>:sample_list/<file>) inventory/<name>.txt`
(the subject finds the commit again after a rebase).

v7 (spec 3.2, 6.11) keeps the names of the KingMaker lists: `inventory/sm_<era>_binned.txt` (data and MC),
`sm_<era>_binned_mc.txt` (the MC of the b-tag probe production P2), `sm_<era>_embedding.txt` and
`sm_<era>_tau_id_measurement.txt` are byte-identical copies of CROWN `sample_list/` (bbtautau `feat-v7-run2`
`0be6596` "feat: add the v7 sample lists of all four Run-2 eras"; `tests/test_inventory.py` compares them with the CROWN
checkout of the workspace when it exists). `sm_<era>_borrowed.txt` is ShapeSmith's own: the lines
`<source era>/<nick>` an era borrows (2016postVFP: ZH, W⁺H and W⁻H → ττ of 2016preVFP, spec 3.3), empty in the other
eras, so that one run YAML lists `sm_{era}_borrowed` in every era.

| Era | binned (data) | binned_mc | embedding | tau_id_measurement | borrowed |
|---|---|---|---|---|---|
| 2016preVFP | 84 (15) | 69 | 15 | 30 | 0 |
| 2016postVFP | 75 (9) | 66 | 9 | 24 | 3 |
| 2017 | 84 (15) | 69 | 15 | 30 | 0 |
| 2018 | 81 (12) | 69 | 12 | 27 | 0 |

The list rules (`tests/test_inventory.py`): data of the era's streams (SingleElectron in 2016 and 2017, EGamma in 2018;
no 2016B ver1), DY in the six pT(Z) bins plus the inclusive M-50 (`npartons == 0`) and M-10to50, W 0J/1J/2J, the
powheg signal κλ = 1 and node_SM, VH → ττ (not in 2016postVFP), exactly one ZZ → 2ν2q sample (ZZTo2Nu2Q_5f, 2018
ZZTo2Q2Nu), TTWJets instead of ttWJets, no ZH → bb with Z → bb (in ZToQQ); embedding: the three tau final states of
every run period; tau-ID lists with W 0J/1J/2J (E17).

`sm2018_binned_v6` drops three samples counted twice: the inclusive LO `ttZJets` (ttZ is `TTZToLLNuNu_M-10`,
`TTZToQQ` and `TTZToLL_M-1to10`), `TTZToNuNu` (contained in `TTZToLLNuNu_M-10`) and `ZZTo2Nu2Q_5f` (the process of
`ZZTo2Q2Nu`), as the legacy analysis. `ttWJets` stays, the only ttW sample of the database. Being a subset of
`sm2018_binned_v3`, it also serves the v5 production (hist, estimate and plot only).

`sm2018_binned_v3` adds the inclusive DY M-50 amcatnloFXFX sample, of which only the `npartons == 0`
events are kept (`SAMPLE_CUTS` in `samples.py`): the LHEFilterPtZ bins lack every zero-parton event
and cover `npartons >= 1`. It needs a production that contains this sample; the v4 run YAMLs keep
`sm2018_binned_v2`, the v5 run YAMLs use `sm2018_binned_v3`. Only this sample reads `npartons`, so the skims of the other samples stay
valid when an existing skim directory switches to `sm2018_binned_v3`.

Routing (`samples.route`): data streams go to the channels whose trigger they carry; embedded samples
(`EmbeddingRun…`) by their final state, `_mutau_` → mt, `_eltau_` → et, `_tautau_` → tt (group `EMB`) and
`_muemb_` → mm (group `MUEMB`, for the tau-ID measurement); every other sample by its nick prefix. A channel
holds only the samples of its processes. The skim contract records `nevents` for every kind. The database
`94bd5a0f` has a wrong mutau 2018D event count (21,362,177; its 57 files hold 25,419,943): use a database with the
corrected count, and re-skim a skim made with the old one once with
`shapesmith skim --force --samples EmbeddingRun2018D_cwinter-embedding_2018UL_mutau`.

`tests/fixtures/datasets.json` holds the database entries of exactly the nicks in `inventory/`, taken from the sibling
checkout `KingMaker_sample_database` at `v7-run2-registration` `0de63ebe` (the v7 registration and the HH-WG 2026 / YR4
mH = 125.00 cross sections; `tests/test_inventory.py` compares them with the sibling checkout when it exists); a new
inventory adds its entries in the same change. The v2-v6 run YAMLs read the same sibling checkout, so their single-Higgs
and HH cross sections are now those of v7 (HH 0.002221 pb instead of 0.002246 pb).

## Run 2 (v7)

Every Analysis builds one era, `era:` of the run YAML (`2016preVFP`, `2016postVFP`, `2017`, `2018`; any other fails),
with the luminosity of `constants.LUMI_PB` (19.52 / 16.81 / 42.07 / 59.56 fb⁻¹, E20; plots `59.6 fb⁻¹ (2018, 13 TeV)`,
Run 2 `138 fb⁻¹`). The 2016 halves stay separate eras up to the datacards (bins `htt_<channel>_<i>_<era>`); the
nuisances they share are named so. Spec: `docs/v7_spec.md` of the workspace, sections 6.1-6.7.

- **Triggers** (`triggers.py`, `TRIGGER[era][channel]`, spec 6.2): flags, offline thresholds and the MC and embedding
  weights per era. Muon 2016 the IsoMu22 OR (pT > 24 GeV, the triggering muon |η| < 2.1 in mm and em), 2017 IsoMu27
  (> 29 GeV), 2018 IsoMu24, each with the KIT SF `trg_wgt_single_mu22/27/24`; electron 2016 Ele25 (> 27 GeV,
  |η| < 2.1), 2017 Ele32 for 34-37 GeV and Ele35 above with the weight piecewise and the HLT Zvtx factor 0.991 above
  37 GeV on MC and embedding, 2018 Ele32; tt 2016 the data with Iso or CombinedIso and the MC with Iso (its POG SF),
  2017 three and 2018 four paths. The tt embedding has no trigger requirement (the cut uses the skim's kind flags
  `is_data`, `is_mc`, `is_embedding`) and the weight ε_data(τ1)·ε_data(τ2), each leg in its isolation class
  (`trg_effdata_emb_{Medium,AntiMedium}_{1,2}`, E33). The tau-ID measurement takes the muon trigger of the era
  (2018: IsoMu24 or IsoMu27 from 25 GeV, as before).
- **Selections**: |η| < 2.1 of the light lepton in et and mt (`lepton_eta`) and of the muon in the tau-ID measurement
  (`muon_eta`), every era (E31, the embedding acceptance); |η| < 2.1 of both taus in tt (`tau_eta`, the acceptance
  of the di-tau trigger matching, which the untriggered tt embedding lacks: about 3 % of its events had a tau at
  2.1 < |η| < 2.5, outside the range of its measured ε_data); in 2018 `hem_veto < 0.5` in every channel of every
  Analysis (E29; the HEM nuisance is gone).
- **Weights**: `prefiring_wgt` on the MC of every era (E11; 2018 is the muon part), not on data or embedding;
  `CMS_prefiring` from the nominal-tree columns `L1PreFiringWeight_Up/Dn`, written as lnN (the varied over the
  nominal yield per bin and process, `Analysis.lnn_variations`). The b-tag weight variations follow the [T, M]
  scheme: `CMS_btag_fixedWP_{bc,light}_correlated` and `..._uncorrelated_<era>` (columns
  `btag_weight_upart_{up,down}_{bc,light}_{correlated,uncorrelated}`). `DY_NJETS_BTAG` is per era and 2016 half
  (E19, spec 6.10), each from the v7 dilepton run of the era: `scripts/derive_dy_correction.py -c <dilepton run>
  -s era=<era> --ttbar-scale em` (`dy_correction.py`) prints the entry of the era.
- **Samples**: SingleElectron data (2016, 2017) go to et and ee; the powheg `GluGluToHHTo2B2T_kl-1p00` is the signal,
  the other κλ points have no process, node_SM is the auxiliary process `HH2B2Tau_nodeSM` (the cross-check) or, in a
  list without powheg (the 2018 productions up to v6), the signal; `ZHToTauTau`, `W±HToTauTau` are the process
  `VH_htt` (VH theory lnN); `TTWJetsTo*`, `WZTo4Q`, `ZZTo2Nu2Q_5f` go to TTV and VV. Every nick must be a sample of the
  run's era (its database `era`). A borrowed sample is the inventory line `<source era>/<nick>`: read from the
  n-tuples of the source era (`Sample.era`), normalised with the run's luminosity, with the lnN
  `CMS_htt_borrowed_<process>_<era>` 1.05; only the borrowings of `samples.BORROWED` are accepted (2016postVFP takes
  VH → ττ from 2016preVFP, spec 3.3). `samples.inventory_nicks` reads an inventory with borrowed lines.
- **Correlations** (`correlations.py`, spec 6.7): `nuisance(name, family, era)` appends the era token of the family:
  none (correlated), the era (2016 halves separate; the copies of 2018 payloads take 2018, K6) or the year (2016
  shared). lnN: the LUM POG Cholesky set `lumi_13TeV_1516_l` / `_151617_l` / `_15161718_l` of the year (MC only);
  `eff_e`, `eff_m` 2 % on MC and, on the embedded sample, 1 % under the same name plus `eff_e_emb`, `eff_m_emb` 1.7 %
  (E27); the single-lepton triggers `CMS_eff_{m,e}_trigger[_emb]_<era>` 2 % on MC and embedding (the KIT SFs have no
  variations); `eff_t_vsJet_<channel>_<era>` and `eff_t_vsLep_<channel>_<era>` (VSe and VSmu together; 2018 token in
  both 2016 halves, whose VSe and VSmu payloads are 2018 copies, K6); the fake-factor shifts `CMS_ff_<key>_<channel>_<year>` and
  `jetFakesNorm_<channel>_<year>`; `CMS_htt_emb_ttbar_<era>`; HH theory per E26 (`QCDscale_HH` +7.25/−21.12 %,
  `PDF_alphas_HH` 2.47 %). The families of CROWN shifts (TES, JES, JER, MET, recoil, the MC vsJet/VSe/VSmu components,
  the shared EMB/MC parts of TES and the tt trigger) come with the shift production; `correlations.py` names them.
- **Fake factors** (spec 6.8, E18; `ff_measurement.py`, `ff_tables.py`): two payload variants, switch
  `ff_processes`: `[QCD, ttbar]` (output `fake_factors/<era>/`, as v6) and `[QCD, Wjets, ttbar]` (`fake_factors_w/<era>/`).
  The W+jets fake factors (et, mt) are measured from data in the DR mt_1 > 70, no b tag, >= 2 jets, isolated lepton,
  opposite sign, minus every other MC and minus QCD from the same-sign version of each region; the non-closures in the
  DR; the DR->SR correction (from no to >= 1 b tag and from high to low mT) from W MC in mt_1 of the baseline;
  fractions QCD, W+jets and ttbar. The tau analysis with `ff_processes: [QCD, Wjets, ttbar]` declares the W+jets keys
  of the friend (`FF_SHIFTS_W`) in et and mt. 2016 is one measurement of both halves (`era: "2016"`, the lists of both
  halves, where a list named with 2016, e.g. `sm_2016_binned` from `sm_{era}_binned`, stands for those of both halves,
  `samples.inventory_names`; the DY table is the luminosity-weighted mean of theirs; every sample reads the n-tuples of its half and is normalised to its luminosity, `Sample.lumi_scale`), one
  payload `…/2016/`; a run per half with the same tables is the cross-check. Tables per era: the 2018 tables with the
  equipopulated bin counts of `PER_ERA_OVERRIDES` (2018 counts times the luminosity ratio) and the edges of
  `PER_ERA_EDGES`, entered from the JSON line of `measure --suggest-binning` of the era; until an era (and the W+jets
  tables in every era) has its edges, `measure` refuses and only `--suggest-binning` runs.
- **Tau-leg trigger efficiency** (E33; `tau_trigger_measurement.py`, core `shapesmith.measurements.tau_trigger`):
  ε_data of the tau leg of the tt trigger for the tt embedding, μτh tag and probe on the tau-ID production P1, per era,
  isolation class (`wp_class`: `Medium`, `VVVLoose_not_Medium`), decay mode (0, 1, 10, 11, all; `dm1011` merges 10 and
  11) and pT bin (`TRIGGER_PT_EDGES` per era and class from `--suggest-binning` on [40, 45, 50, 60, 80, 120, ∞):
  merged from the top until the statistical error of ε_data, counted with mT < 30 GeV and 40 < m_vis < 90 GeV, is
  at most 0.03; the anti-isolated class is one bin per decay mode): pass/fail m_vis fit
  with the efficiency as POI (1D profile scan), counting cross-check (mT < 30, 40 < m_vis < 90), ε_MC and ε_EMB^mon
  (2016 and 2017 only: the 2018 μτh embedding has no monitoring-path decisions, so 2018 has no monitor process).
  The pass definition follows the tt paths leg by leg with the monitoring columns `trg_mon_*_2`, `trgobj_pt_2`,
  `trgobj_l1pt_2` of CROWN C13/12.4; in 2018 the data before run 317509 pass with the non-HPS legs, the later data, MC
  and embedding with the HPS path (the HPS monitoring path fires in the data from the first 2018 run). The embedded
  taus carry the energy scale of the era's tau-ID/ES payload (`emb_es_payload`, its ±1σ a shape nuisance). Variants
  (`variant`: `signal_mc`, `tag_pt`, `mt30`, `monitor`) are merged as uncertainties; `--merge` writes
  `ditau_trigger_DeepTau2018v2p5_embedding<era>UL.json.gz` (`tau_trigger`, the KIT/POG schema, flow clamp), only
  with a nominal measurement of both classes, which CROWN books, and after evaluating eff_data of both for every decay
  mode and syst. The signal shapes are the ES nuisance and the ttbar
  contamination `CMS_emb_ttbar_contamination_Run<era>` (a template shift, which the core check accepts as an estimator
  output). The ε(pT) plots draw the POG v15 eff_data and eff_mc (`reference_payload`) and the KIT embedding file's
  eff_data and eff_emb (`kit_reference_payload`, CROWN `data/embedding/tau_trigger<era>_UL.json.gz`, DeepTau2017v2p1):

  ```bash
  for cls in Medium VVVLoose_not_Medium; do for variant in nominal signal_mc tag_pt mt30 monitor; do   # no monitor in 2018
    shapesmith measure -c <run YAML> -s era=2017 -s switches.wp_class=$cls -s switches.variant=$variant; done; done
  shapesmith measure -c <run YAML> -s era=2017 --merge
  ```
- **Tau-ID/ES** (spec 5.4, U11): best value and 1σ interval of each POI from its profile, rescans on a widened range,
  the singles fit as a cross-check (`report.md` flags deviations > 20 % of the width), postfit at the profile minima;
  `shapesmith measure --refit` repeats the fits from the stored `synced/` shapes; the payload's pT-binned ES changes
  linearly across 36-42 GeV (no gap in the corrected pT at 40 GeV).
- **Combination** (spec 6.9): `shapesmith combine -c <run of each era> ... -o <dir> [--background-only]` combines the
  cards of a final state of every era with `combineCards.py` and runs the limits or the background-only fit;
  `nuisances.md` flags nuisances that end in another era's token.
- **Tests**: `tests/test_run2.py` builds every Analysis in every era on the v7 inventories (`tests/run2.py`) and checks
  every column of every sample against the branch fixture of its era and sample type (`tests/helpers.py`:
  `branches(channel, era, sample_type)`, `tau_id_branches(...)`); `tests/test_inventory.py` the v7 lists per era and
  their copies; `tests/test_v7_runs.py` every v7 run YAML in every era, its paths, productions and gallery column;
  `tests/test_ff_variants.py`, `tests/test_tau_trigger_measurement.py` and `tests/test_dy_correction.py` the
  fake-factor variants and eras, the trigger measurement and the DY table.

## Embedding and fake factors

- **Processes.** Gen-match splits T (genuine ττ, the lepton leg from a τ decay), L (lepton fakes), J (jet → τh) for
  DY, TT, ST, VV, TTV and EWK. In an embedded channel (`embedding: true`, or the channel in the list
  `embedding: [et, mt]`), `EMB` (genmatch T) replaces every T part; `TTT` stays as an auxiliary process (booked in
  nominal, in no datacard or plot) for the ttbar contamination `CMS_htt_emb_ttbar_2018` (EMB ± 10 % of TTT). The
  lnN table and the ML labels take the processes of all channels. With `jet_fakes: ff`, `jetFakes` replaces the J
  parts and W.
- **EMB weights** (Part-A contract): `emb_genweight`, the embedding selection SFs, the embedding lepton ID/iso SFs
  (mt `id_wgt_mu_1`, `iso_wgt_mu_1`; et `id_wgt_ele_1`, `iso_wgt_ele_1`), the vsJet SF of genuine τh, vsE/vsMu and the
  trigger SF; no pileup, b-tag or electron reconstruction weight (the et MC carries `id_wgt_ele_1` and
  `reco_wgt_ele_1`). They carry their own names (`emb_…`), so regions that replace MC weights (the tau-fail regions,
  the mt muon anti-isolation region) leave them unchanged. In the anti-isolated regions genuine taus keep the Medium
  pass SF, as in TauFakeFactors and the legacy analysis.
- **Shifts** (`systematics.py`, only with `shape_systematics: true`). The analysis declares every CROWN shift it
  reads. For a sample kind with declared shifts the skim fails on a declared shift without a shifted branch and on an
  undeclared `c__X` branch of a needed column (main n-tuple or friend):
  - embedding (only in the embedded channels): `CMS_scale_t_emb_dm{0,1,1011}_<wp>_2018` (branches
    `__CMS_scale_t_emb_DeepTau2018v2p5_DM<dm>_2018`) and `CMS_eff_t_emb_dm{0,1,1011}_<wp>_2018`
    (`__CMS_eff_t_emb_DeepTau2018v2p5_VSjet_DM<dm>_2018`), `wp` = `vsEleTight` in et, `vsEleVVLoose` in mt and tt;
    6 nuisances per channel. The energy scale and the vsJet SF of the embedding are binned in the decay mode only
    (production v6); the v5 embedding has pT-binned vsJet shifts (`..._pt{20to40,40toInf}_2018`) instead, so v5 runs
    with embedding shifts need `--skip-systematics` or `shape_systematics: false`. DM10 and DM11 are one fitted
    category, which CROWN shifts per decay mode: the `dm1011` nuisances are `VariationSum`s of the DM10 and DM11 shifts (read as the
    parts `<name>Up%dm10`, ...), exact in et and mt, to first order in tt, where both taus can be 3-prong.
  - fake factors (only with `jet_fakes: ff`): `CMS_ff_<key>_<channel>_2018` with branch suffix `__<key>`, for the
    17 keys of `FF_SHIFTS_LT` (et, mt) and the 34 of `FF_SHIFTS_TT` (tt, both legs), on data, MC and embedding.
    They follow the SM 2018 payload; a new payload with other keys fails the skim until the lists follow it.
  - `jetFakes` is built for every shift found on data or a subtracted process in `anti_iso`, so the embedding shifts
    reach it too; the sums run after it.
- **One skim for all switches.** A skim made with `embedding: true, jet_fakes: ff` (FF friend configured) serves all
  four combinations. Skims made without the friend or the shifts are incompatible with it: the first such run needs
  `shapesmith skim --force --samples …` for the samples it lists.
- **Raw fake factors** (`ff_type: raw`, `configs/sm2018_binned_v5_rawff.yaml`). `jetFakes` is weighted with the raw FF
  of the friend (`fake_factor_raw` in et and mt, `fake_factor_1_raw`/`fake_factor_2_raw` in tt with the same half per
  failing leg): the fractions times the QCD and ttbar fake factors, without the DR->SR and non-closure corrections.
  A diagnostic of the corrections with nominal control plots only: the raw ttbar FF still carries the global ttbar
  data/MC factor of the measurement (`ttbar_data_scale`: et 0.70, mt 1.15, tt 0.39 leading / 0.89 subleading), which
  the first ttbar non-closure cancels in the corrected FF. The raw FF columns carry only the 8 shift keys of the
  fractions and fake factors, so `shape_systematics: true` is rejected. Its skim holds the raw instead of the corrected
  columns and has its own `skim_dir`.

## Fake-factor measurement

`configs/ff_sm2018_binned_v6.yaml` runs `bbtautau_shapesmith.ff_measurement:build`, the SM 2018 fake-factor
measurement in jvoss's TauFakeFactors method (QCD and ttbar fake factors, fractions, the QCD DR->SR correction and the
non-closures), with its own skim (both charges, the lepton vetoes stored for the ttbar scale regions). It derives the
payload of the v6 friend ff_v2 (embedding in et and mt, MC in tt); `ff_sm2018_binned_v6_noemb.yaml` is its
cross-check without embedding on the same skim. `ff_sm2018_binned_v5.yaml` (payload of ff_v1, embedding in every
channel) and `ff_sm2018_binned_v4.yaml` (without embedding) are the earlier measurements.

```bash
shapesmith skim    -c configs/ff_sm2018_binned_v6.yaml
shapesmith measure -c configs/ff_sm2018_binned_v6.yaml [--suggest-binning]   # -> output/.../fake_factors/2018/
```

Switches of the measurement (`FakeFactorSwitches`): `sample_lists`; `embedding` as in the analysis (`false`, `true`
or the embedded channels: genuine τh subtracted as EMB there and as the MC T parts elsewhere, so `[et, mt]` gives the
mixed payload of v6 in one run); `dy_correction` as in the analysis, on the subtracted DY processes;
`tt_fraction_region`: `both_anti_iso` (default, U6 and jvoss parity) measures the tt fractions of both legs with both
taus anti-isolated, `application` those of each leg in its application region (that tau anti-isolated, the other
isolated), where ttbar is 3-6 times more frequent (v6).

The regions are built in `ff_measurement.py` from the analysis selection (`cuts.py`); every difference to jvoss's
configuration is listed in its docstring. The binning, fits and bandwidths are `ff_tables.py`: the 2018 tables generated once by
`scripts/make_ff_tables.py` from TauFakeFactors' resolved configuration (do not edit them by hand), below its marker line
the hand-written W+jets tables and the per-era tables (v7, see Run 2), which the script keeps. `scripts/ff_parity.py`
is the algorithm parity with TauFakeFactors: it runs the measurement on jvoss's preselection files with his regions
and weights and compares every payload value, edge and fitted curve with his payload and his pickled fits, and
`--suggest-binning` with his `adjust_binning.py`. One explained difference is by decision (U13): the SystMCShiftDown
of the corrections is smoothed at the centres of mass; TauFakeFactors smooths it at the bin centres because of a
deepcopy bug, which the parity reproduces separately. A new payload goes into a new dated directory of the CROWN
analysis (`payloads/fake_factors/sm/fake-factors-<date>/`) and needs a fresh friend tag; `constants.FF_SHIFTS_LT/TT`
must follow its keys.

## Tau-ID and ES measurement of the embedded taus

`tau_id_measurement.py` (`configs/tau_id_es_2018.yaml`, per era since v7) is another measurement Analysis: the successor of smhtt_ul
`tauID_SFs_dev`, run by the core measurement `shapesmith.measurements.tau_id_es`. It needs the production of CROWN
`sm_tau_id_measurement_config` (list `sm2018_tau_id_measurement`, scopes mt and mm, all shifts): production
`sm2018_tau_id_v1` (2026-10-03/04, 0.73 TB, mostly MC with all shifts).

- **Selection** as smhtt_ul `config/shapes` (special `TauID_ES`, 2018) with bbtautau column names: mt tag and probe
  with `mt_1 < 65`, `IsoMu24 || IsoMu27` and pT > 25 / 20 GeV, the nine categories DM0, DM1, DM1011 (pT >= 20 GeV) and
  their [20, 40) and [40, 200] GeV bins; the mm control region 70-110 GeV in one bin. The m_vis bins of the mt
  categories are ten bins with equal data counts in 30-160 GeV, as smhtt_ul `gof/build_binning.py`
  (`tau_id_binning.py`): ShapeSmith computes them from the OS data of the working-point combination in each category
  whenever it fills, and records them in `binning.json` next to `shapes.root`
  (`<output_dir>/tau_id_es/2018/<vsjet>_<vsele>/`).
- **Processes.** mt: EMB (genmatch 4/5; no tau ID, vsEle, vsMu or ES correction), ZL, ZJ, TTL, TTJ, STL, STJ, VVL, VVJ,
  W and QCD = same-sign data minus all of them (negative bins clipped); TTT is the auxiliary template of
  `CMS_emb_ttbar_contamination_Run2018`. mm: MUEMB (genmatch 2/2, muon SFs of both legs), W, TTL, VVL and QCD =
  same-sign data minus MUEMB and W. MC and embedding weights are the predecessor's: KIT muon ID/iso SFs, the trigger
  weight `mu24` for 25 <= pT < 28 GeV and `mu27` above 28 GeV in 2018 (2016 and 2017: the trigger of the era, see Run 2),
  the prefiring weight on the MC. The MC is normalised to the luminosity of the era.
- **Switches** `vsjet_wp` (Medium, Tight) and `vsele_wp` (VVLoose, Tight). The skim takes the loosest of them, and the
  four combinations are regions `wp_<vsjet>_<vsele>`, so one skim serves all four runs.
- **ES grid**: `es-200` ... `es+200` (0.2 % steps, units of 0.1 %), template column variations of the embedded sample
  in the nominal region, derived from the nominal columns: the tau four-momentum scaled (pt_2, mass_2, m_vis from the
  four-vectors), the MET corrected by (1 - s) times the tau pT, mt_1 from that MET.
- **MC shape uncertainties** (`tau_id_systematics.py`): the predecessor's families with MorphingTauID2017's names, from
  the CROWN shifts `<quantity>__<shift>` of `sm_tau_id_measurement_config` (variations package) and the jet-fake and
  top-pT weights; mt only, nominal region only. `CMS_scale_t_dm*` (three pT bins each) and `CMS_scale_fake_m` (five
  wheels) are VariationSums of the finer CROWN shifts; the jet energy scale is the regrouped set (no HEM since v7);
  `CMS_prefiring` from the CROWN shift `CMS_ecal_prefiring_<year>`; the names carry `Run<era>`. Produced shifts without a
  shape in MorphingTauID2017 (vsEle SF, electron -> tau ES, JER, electron ES as `CMS_scale_e`/`CMS_res_e`) are filled
  under their own name and ignored by the datacards. `CMS_htt_ttbarShape` is the difference with (the nominal)
  against without the top-pT weight (K9): Down without it, Up the mirror 2 w - 1 around the nominal. The vsJet SF families `CMS_eff_t_dm*` equal the nominal, as in the predecessor.
- **CMSSW**: MorphingTauID2017 exists only with jvoss's local changes plus the regrouped JES declaration and the v7
  changes (no HEM, `CMS_prefiring` in every era, the Cholesky lumi total), versioned in `patches/`; it needs an own
  area built with the patch (see `patches/README.md`). `tests/test_tau_id_patch.py` checks per era that every shape the
  patched file declares is produced.

## Where things are defined

| What | Where |
|---|---|
| produced samples (verbatim KingMaker sample lists), groups, routing, per-sample cuts, normalisation | `inventory/*.txt`, `samples.py` |
| the switches and their checks | `switches.py` |
| eras, luminosity, channel sets, working points, b-tag and jet bins, friend columns, class names, FF shift keys, embedding shift tables, the DY correction table, HEM, lepton eta, prefiring | `constants.py` |
| triggers per era and channel: cuts, thresholds, MC and embedding weights | `triggers.py` |
| correlation of the nuisances between the eras | `correlations.py` |
| tau channels: baseline/skim selection, gen-match splits, estimation and control regions | `cuts.py` |
| MC, embedding, fake-factor and dilepton weights, trigger chains, the DY correction weight | `weights.py` |
| process tables (tau: gen-match split, dilepton: unsplit) | `processes.py` |
| NN categories, control variables and binning | `variables.py` |
| b-tag weight variations, embedding and fake-factor shifts, lnN table | `systematics.py` |
| colours, labels, axis titles of all six channels | `style.py` |
| tau analysis assembly (with the control channel em), estimators, ML export | `analysis.py` |
| fake-factor measurement: regions, processes, legs (QCD, W+jets, ttbar); its tables per era | `ff_measurement.py`, `ff_tables.py` |
| tau-leg trigger efficiency for the tt embedding: selection, pass definition, bins, ES, variants | `tau_trigger_measurement.py` |
| the DY table of an era from its dilepton run | `dy_correction.py`, `scripts/derive_dy_correction.py` |
| dilepton channels and analysis assembly; the em control channel of the tau analysis | `dilepton.py` |
| tau-ID/ES measurement: selection, processes, ES grid, assembly; its m_vis bins; its MC shape uncertainties | `tau_id_measurement.py`, `tau_id_binning.py`, `tau_id_systematics.py` |

`tests/fixtures/branches/<era>.json` hold, per era, configuration (`sm_config`, `sm_tau_id_measurement_config`),
scope and sample type, the branches CROWN's generated program writes (`scripts/branches_from_crown.py` over the
generate.py dry run of CROWN bbtautau `feat-v7-run2` cfcca94; the programs the branch cannot generate yet, 2016 tt and
the embedding and tau-ID production outside 2018, from a copy with the C8 dict and the era guards patched; ttbar in
full, the other sample types as differences). For 2018 ttbar they are the branches of real `sm2018_binned_v2` files
(`tests/fixtures/branches/sm2018_binned_v2/<channel>.txt`, `scripts/dump_branches.py`) plus the six v6 columns. The tests
check every selection, weight and variable of every sample against the fixture of its era and sample type plus the v7
CROWN output that branch does not write yet (`tests/helpers.py`, `branches(channel, era, sample_type)` and
`tau_id_branches(...)`: the trigger columns of the era, prefiring, the [T, M] b-tag variations, the 2018 HEM flag, the
tt embedding's di-tau efficiencies, the trigger monitoring columns of the tau-ID production; the `V7_*` sets, the
counterpart of the CROWN tests' PENDING items). With the CROWN parts C6-C8, C11-C14, C20 merged, the fixtures are
written again and the `V7_*` sets shrink to nothing; with the first v7 files `dump_branches.py` checks them.
`tests/fixtures/skim_contracts_v4.json` holds the stored contracts and Parquet columns of one data and one MC skim per
channel of v4 (from the golden references of 2026-09-30); `tests/test_contracts_v4.py` checks that the v4 run YAMLs
differ from them by exactly the changes since (so a v4 skim needs `skim --force`).

## Prerequisites outside this repository

- NN friend trees (`predicted_class`, `predicted_max_value`) from a trained model + CROWN `sm_ml.py`. With embedding
  shifts the NN friend must be produced with `--shifts all`, or the NN category and score stay nominal under them.
- Fake-factor friends (`fake_factor`, `fake_factor_1/2`, the raw `fake_factor_raw`, `fake_factor_1/2_raw` and their
  shifts) from CROWN `fake_factors_friend_config.py`, on data, MC and embedding. A new FF payload gets a new friend tag
  (a new `ntuples.friends` base); the skim contract records the friend bases, so the stored skims then need `skim --force`.
- Embedding ntuples under the same `ntuples.base` and production tag as the data/MC they are combined with, produced
  with `--scopes et,mt,tt --shifts all`.

Until they exist: `nn_friend: false`, `jet_fakes: mc`, `embedding: false`.
