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
shapesmith hist      -c configs/sm2018_binned_v6_ff.yaml && shapesmith estimate -c configs/sm2018_binned_v6_ff.yaml   # with systematics (the card shapes); categories nominal, em_tt
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
# the NN training data: see "NN training data (ML export)"; the NN categories and limits: "NN categories, limits and blinding"
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
# P3: dilepton controls and the DY table of the era (constants.DY_NJETS_BTAG; v7b: smrun2_binned_v7b_dilepton.yaml)
D=configs/smrun2_binned_v7_dilepton.yaml
shapesmith skim -c $D $S && shapesmith hist -c $D $S --control --regions all --skip-systematics && shapesmith plot -c $D $S --control
python scripts/derive_dy_correction.py -c $D $S --ttbar-scale em
# tau control runs, columns emb and classic (one skim)
T=configs/smrun2_binned_v7.yaml; N=configs/smrun2_binned_v7_noemb.yaml
shapesmith skim -c $T $S
for C in $T $N; do shapesmith hist -c $C $S --control --regions all --skip-systematics && shapesmith estimate -c $C $S --control && shapesmith plot -c $C $S --control; done
# fake factors, both variants (2016: -s era=2016, both halves; the halves alone are the cross-check)
shapesmith skim    -c configs/ff_smrun2_binned_v7_w.yaml $S
shapesmith measure -c configs/ff_smrun2_binned_v7_w.yaml $S --suggest-binning   # edges -> ff_edges.py (scripts/ff_binning.py)
shapesmith measure -c configs/ff_smrun2_binned_v7.yaml $S && shapesmith measure -c configs/ff_smrun2_binned_v7_w.yaml $S
# FF runs (after the friends ff_v7, ff_v7_w), datacards, background-only fit, then the Run-2 combination
for F in configs/smrun2_binned_v7_ff.yaml configs/smrun2_binned_v7_ff_w.yaml; do
  shapesmith skim -c $F $S
  shapesmith hist -c $F $S --control --skip-systematics && shapesmith estimate -c $F $S --control && shapesmith plot -c $F $S --control
  shapesmith hist -c $F $S && shapesmith estimate -c $F $S && shapesmith datacards -c $F $S && shapesmith fit -c $F $S --background-only   # hist with systematics: the card shapes
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

Every era and 2016 half has its DY table (`constants.DY_NJETS_BTAG`, from its `smrun2_binned_v7b_dilepton.yaml` run
since 2026-10-19; the v7 tables came from `smrun2_binned_v7_dilepton.yaml`); the 2016 FF measurement takes the
luminosity-weighted mean of the tables of both halves. `--workers` stays at most 24 on portal2; the dilepton `hist`
needs up to ~45 GB per worker for the DY and data samples of 2017 and 2018 (57 GB with one worker in 2017 v7b).

**v7b (re-production, 2026-10-18).** The MET fix (DY and W recoil-corrected in 2018 only, on the PuppiMET with the
leptons before the jet energy scale; 2016 and 2017 without recoil correction) and the POG muon_Z trigger SF on MC (E5)
need new MC and embedding: P3 MC and embedding as `smrun2_binned_v7b` (its friends `fastmtt` and `ff_v7b`, `ff_v7b_w` of the
MC and embedding there too; the FF friends have their own names, so that P5 of v7 and v7b never overwrite each other,
and their data friends lie under `smrun2_binned_v7`), P1 MC as `smrun2_tau_id_v2`; the data (P1: and the embedding) stay. Every v7 run YAML has
its copy that reads the re-produced kinds from the new tag (`ntuples.bases`, friends split by `applies_to`) and writes
skims, outputs and galleries under the new tag, so the v7 run YAMLs keep reproducing the v7 results
(`tests/test_v7_runs.py` checks that each copy is its v7 run otherwise): `smrun2_binned_v7b.yaml`, `_noemb`,
`_dilepton`, `_ff`, `_ff_w`, `ff_smrun2_binned_v7b.yaml`, `ff_smrun2_binned_v7b_w.yaml`, `smrun2_tau_id_v2_id_es.yaml`
(`recoil_eras: ["2018"]`; the v1 YAML sets every era) and `smrun2_tau_id_v2_trigger.yaml` (the ES payload of
`smrun2_tau_id_v2`). The commands are those above with these YAMLs. Their galleries are the eras of two sites
(`web.site`), one page each with an era switch: https://web.etp.kit.edu/~sdaigler/smrun2_binned_v7b/ and
https://web.etp.kit.edu/~sdaigler/ff_smrun2_binned_v7b/ (with 2016, the measurement of both halves); the commands
below add or replace an era. `constants.DY_NJETS_BTAG` holds the tables of the
`smrun2_binned_v7b_dilepton.yaml` runs (2026-10-19); they serve the v7 run YAMLs as well, so a v7 run with
`dy_correction: njets_btag` no longer reproduces its v7 result exactly (the v7 MC is superseded; the v7 tables are in
the git history, analysis abd7657).

The v7b tau runs (`smrun2_binned_v7b.yaml`, `_noemb`, `_ff`, `_ff_w`) read the FastMTT friend of P4 (data under
`smrun2_binned_v7`, MC and embedding under `smrun2_binned_v7b`; switch `fastmtt_friend`, since 2026-10-20), so their
control plots include m, pt, eta and phi of the di-tau system from FastMTT.

tt has no embedding in v7b (E33: genuine ττ from MC), so as in v6 the control site shows tt only in the columns classic
and ff (the tt of the FF run; the same in `ff_v7b_w`, which has W+jets fake factors in et and mt only), and emb, emb ff
and emb ff with W hold et and mt (the emb ff columns also em). Publish a column again with fewer channels and its other
channels stay: to drop a channel, remove the era directory first and publish the era anew.

```bash
S="-s era=$E"; B=configs/smrun2_binned_v7b        # E: 2016preVFP, 2016postVFP, 2017, 2018 (FF measurement also 2016)
shapesmith publish -c ${B}_noemb.yaml $S            # ~sdaigler/smrun2_binned_v7b/<era> (+ site page), column classic: et, mt, tt
shapesmith publish -c ${B}_dilepton.yaml $S         # column classic: em, mm, ee
shapesmith publish -c $B.yaml $S --channels et,mt   # column emb
shapesmith publish -c ${B}_ff.yaml $S --channels et,mt,em     # column emb ff
shapesmith publish -c ${B}_ff_w.yaml $S --channels et,mt,em   # column emb ff with W
shapesmith publish -c ${B}_ff.yaml $S --channels tt --variant ff --label ff \
                   --description "MC genuine ττ + jet→τh fakes from the fully corrected fake factor (friend ff_v7b; the same in ff_v7b_w, which has W+jets fake factors in et and mt only); tt only, it has no embedding."   # column ff: tt
shapesmith publish -c configs/ff_smrun2_binned_v7b.yaml $S && shapesmith publish -c configs/ff_smrun2_binned_v7b_w.yaml $S   # ~sdaigler/ff_smrun2_binned_v7b
```

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
| `nn_prefix` | `""` (default), a column prefix | the model of the run among the models of the NN friend: it reads `<prefix>predicted_class` and `<prefix>predicted_max_value` (categories, control variable `NN_score` without the signal category, see the blinding rules) |
| `nn_friend_prefixes` | `[]` (default), the prefixes of the friend | every model of the friend production; the skims keep the class and score columns of each, so the runs of all its models share one skim (must contain `nn_prefix`; empty: `nn_prefix` only) |
| `nn_classes` | none (default), a list | the classes of the model in the order of `predicted_class`, with `HH2B2Tau` once (none: those of `ml_labels`, else `thesis6` with `jet_fakes: mc` and `v7b5` with `ff`); K = their number |
| `nn_manifest` | none (default), a path | the `manifest.json` of the model's CROWN payload (`{channel}` and `{era}` filled), whose `classes` must equal the run's classes in order, else the build fails |
| `nn_categories` | `k1` (default), `k2`, `thesis` | the NN categories (`variables.py`, spec 6.2): the argmax without `n_bjets >= 3` (K1), K1 with the HH category split into `HH2B2Tau_btag1` and `HH2B2Tau_btag2` (K2), or the argmax of the thesis (R3-R5) |
| `nn_binning` | `data_blind` (default), `thesis` | their binning, resolved by `datacards` from 1000 fine bins: the data-blind rules of spec 6.3, or the custom edges and AutoRebin of the thesis |
| `fastmtt_friend` | `false` (default), `true` | the control variables `m_fastmtt`, `pt_fastmtt`, `eta_fastmtt` and `phi_fastmtt` of et, mt and tt from the FastMTT friend (CROWN `sm_fastmtt`, tag `fastmtt`; em has none), which `ntuples.friends` must list for the tau channels |
| `fit_categories` | `nn` (default), `control`, `none` | the categories of `hist`, `datacards` and `fit`: the NN classes (needs `nn_friend: true`, else the run fails: without the friend there would be no tau category and cards with em only), one category `nominal` per tau channel with the nominal region in one bin (`yield`), e.g. for `fit --background-only` without NN friends, or none (a run of control plots) |
| `ml_labels` | none (default), `thesis6`, `v7b5` | the ML export and its label profile (`ml.py`, see "NN training data"): `thesis6` needs `jet_fakes: mc`, `v7b5` `jet_fakes: ff`; the class order of the profile is also that of the NN categories (`predicted_class`) unless `nn_classes` names them |
| `ml_variables` | none (default: every set), a list | the export variables: column names or the sets `thesis`, `objects`, `kinematics`, `reskim`, `truth` of `ml.py`; the skims of the run hold them (`reskim` and `truth` need the re-skim of NN spec X8); needs `ml_labels` |
| `signal_sample` | `powheg` (default), `node_sm` | the signal `HH2B2Tau` from the powheg κλ = 1 sample, or from the madgraph node_SM sample of the master's thesis (R3, N9), the powheg sample then left out |
| `control_regions` | `false` (default), `true` | wider skim plus the named pass/fail control regions; needs `jet_fakes: mc` |
| `diagnostics` | `false` (default), `true` | QCD in the control regions (et/mt: same-sign data minus MC, tt: ABCD with the fail regions) and the variables `met_uncorrected`, `metphi_uncorrected`, `npv_good`, `rho` (the last two from production v6 on); needs `control_regions: true` |
| `shape_systematics` | `true` (default), `false` | the CROWN shifts of the embedded samples and of the fake-factor friend as shape variations |
| `crown_shifts` | `true` (default), `false` | `false` for a production without CROWN shifts (v7 is nominal only, E34): no embedding tau shifts; the fake-factor friend shifts and the weight variations of the nominal tree (b tag, prefiring) stay |
| `dy_correction` | `none` (default), `njets_btag` | every DY process (ZTT, ZL, ZJ) times the DY data/MC table `DY_NJETS_BTAG` (`constants.py`) of its n_bjets (0, 1, ≥ 2) and n_jets (≤ 2, 3, ≥ 4) |
| `ttbar_shape` | `false` (default), `true` | stage S0+ of the NN phase (SH7): the top-pT shape `CMS_htt_ttbarShape` of ttbar, see "Systematics stages" |
| `heavy_flavour` | `false` (default), `true` | stage S0+ (SH8): `CMS_htt_heavy_flavour`, 30 % on the heavy-flavour component at ≥ 3 b tags; reads `bpair_hadronFlavour_{1,2}` (skims of the re-skim X8 or later) |
| `signal_theory` | `[]` (default), a list of `qcd_scale`, `parton_shower`, `pdf` | stage S1 (SH14): the signal theory shapes normalised to the inclusive rate; needs `crown_shifts: true` (the shift production P6) and `signal_sample: powheg` |
| `nn_validation` | `false` (default), `true` | the input validation of the NN phase (see "NN input validation"): the control variables are the NN inputs of the channel in fine bins, their eta-phi occupancy cells and `yield`; the skims keep the inputs of FastMTT |
| `pseudo_shifts` | `false` (default), `true` | PRELIMINARY shape nuisances of the input validation: jets ±3 %, tau_h ±1.5 % with the MET propagated and the derived quantities recomputed (`CMS_pseudo_jes_<era>`, `CMS_pseudo_tes_<era>`, `CMS_pseudo_emb_tes_<era>`), symmetrised in the cards; with the NN friend also of its outputs (NV9 in S0+: the friend columns `<column>__<shift>` of `scripts/pseudo_nn_friend.py --pseudo-shifts`); needs `nn_validation` or `nn_friend` |
| `ff_transfer` | `false` (default), `true` | the region `anti_iso` with the fake factor of the friend and its shifts (with the decay-mode non-closure of the fake factors) in `mc` mode (MC jet fakes in AR x FF against the nominal region); needs `jet_fakes: mc` and the fake-factor friend |
| `nn_outputs` | `false` (default), `true` | the validation of the NN outputs (see "NN output validation"): the control variables are the outputs of the model (`NN_cat_<category>`, `NN_P_<class>`, `n_jets` and `mass_2` per category, `NN_ar_<HH category>`; with `ff_transfer` the MC-only copies `NNmc_*`) and `yield`; needs `nn_friend` |
| `nn_blinding` | path with `{era}` | the `blinding.json` of the run's datacards: its edges show the `lo` part of the HH categories in the nominal region (without it they hide every event there); needs `nn_outputs` |
| `nn_ff_payload` | path with `{year}`, `{channel}` | the fake-factor payload of the friend: the regions `anti_iso_ttbar_fraction(_up)`, the anti-isolated region weighted by its ttbar fraction (NV4b); needs `nn_outputs` |
| `ff_same_sign` | `false` (default), `true` | the jet fakes of the same-sign region from its anti-isolated part times the fake factor (region `same_sign_anti_iso`); needs `jet_fakes: ff` |

**em as the ttbar control channel.** A tau run YAML can list `em` in `channels` (e.g. `[et, mt, tt, em]`). It is a
control channel (`Analysis.control_channels`): `datacards` and `fit` put it into every final state, never into one of
its own. It is the dilepton em selection (OS, isolated e and μ, ≥ 2 jets) with the unsplit MC of the dilepton
analysis (ttbar with the top-pT weight), one category `em_tt` with the b tag of the tau baseline counted in one bin
(with NN categories two, `em_tt_btag1` and `em_tt_btag2`, without `n_bjets >= 3`, SH4), no further regions, the b-tag and prefiring variations of the tau channels and the lnN `eff_e`, `eff_m` and
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

Repeated events (decided 2026-10-26): the tau, dilepton, tau-ID and trigger analyses set `event_key` = (run, lumi,
event) (`constants.EVENT_KEY`), so every data and embedding sample is read without the skim rows that repeat an event of
an earlier row of the sample, in file order (core README, "Steps"); the skims keep every row, and `hist`, `measure` and
`ml-export` list the dropped rows in `repeated_events.json`. The only case in v7b is the mutau 2018D embedding: its
input files `nano_35` and `nano_53` share 44 497 events, which leaves 59 repeated rows in the `tau_ff`/`tau_ff_nn`
skims (2018 mt) and 243 in `tau_control`; every other data and embedding sample of the `tau_ff_nn`, `tau_control` and
`dilepton` skims has none. The fake-factor measurement keeps them: its skims hold no run and lumi.

`tests/fixtures/datasets.json` holds the database entries of exactly the nicks in `inventory/`, taken from the sibling
checkout `KingMaker_sample_database` at `feat-run2-nanoaodv15-samples` `0de63ebe` (the v7 registration and the HH-WG
2026 / YR4 mH = 125.00 cross sections; `tests/test_inventory.py` compares them with the sibling checkout when it
exists); a new inventory adds its entries in the same change. The v2-v6 run YAMLs read the same sibling checkout, so
their single-Higgs and HH cross sections are now those of v7 (HH 0.002221 pb instead of 0.002246 pb).

## Run 2 (v7)

Every Analysis builds one era, `era:` of the run YAML (`2016preVFP`, `2016postVFP`, `2017`, `2018`; any other fails),
with the luminosity of `constants.LUMI_PB` (19.52 / 16.81 / 42.07 / 59.56 fb⁻¹, E20; plots `59.6 fb⁻¹ (2018, 13 TeV)`,
Run 2 `138 fb⁻¹`). The 2016 halves stay separate eras up to the datacards (bins `htt_<channel>_<i>_<era>`); the
nuisances they share are named so. Spec: `docs/v7_spec.md` of the workspace, sections 6.1-6.7.

- **Triggers** (`triggers.py`, `TRIGGER[era][channel]`, spec 6.2): flags, offline thresholds and the MC and embedding
  weights per era. Muon 2016 the IsoMu22 OR (pT > 24 GeV, the triggering muon |η| < 2.1 in mm and em), 2017 IsoMu27
  (> 29 GeV), 2018 IsoMu24, each with the SF `trg_wgt_single_mu22/27/24` (E5: on MC the POG muon_Z SF from
  smrun2_binned_v7b and smrun2_tau_id_v2 on, the KIT SF in v7 and tau_id_v1; on the embedding the KIT SF); electron 2016 Ele25 (> 27 GeV,
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
  (E19, spec 6.10), each from the v7b dilepton run of the era: `scripts/derive_dy_correction.py -c <dilepton run>
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
  (E27); the single-lepton triggers `CMS_eff_{m,e}_trigger[_emb]_<era>` 2 % on MC and embedding (the SFs of the nominal
  production have no variations); `eff_t_vsJet_<channel>_<era>` and `eff_t_vsLep_<channel>_<era>` (VSe and VSmu together; 2018 token in
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
  equipopulated bin counts of `PER_ERA_OVERRIDES` and the edges of `PER_ERA_EDGES` (both in `ff_edges.py`, written by
  `scripts/ff_binning.py`); until an era (and the W+jets tables in every era) has its edges, `measure` refuses and
  only `--suggest-binning` runs. The v7b binning (2026-10-19): `ff_binning.py counts <output_dir>` reviews the
  measured ratios of the W variant (median relative stat error per category above 0.30 lowers a count, in 2016 and
  2017 one below 0.25 raises it up to the 2018 count; stable after two rounds), `ff_binning.py edges <measure log>
  --era <era>` enters the `--suggest-binning` edges. 2018 keeps jvoss's measured QCD and ttbar tables except the
  reviewed quantities (QCD DR->SR of et and mt with 2 bins, et QCD iso_1 non-closure), the mt lepton pT non-closures,
  which start at the trigger threshold of the era (`PER_ERA_LOW`: 24, 29, 25 GeV instead of 32), and the ttbar fake
  factors of et and mt, smoothed with 6 GeV instead of 26/12 GeV (`FIT_OVERRIDES`). `scripts/ff_validation.py` checks
  a payload on its skim: the DR closure of the fully corrected fake factor in every table variable and the jet-fake
  prediction per b-tag and jet category (both variants compared on it).
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
  under their own name and ignored by the datacards. The recoil calibration `CMS_scale_met`/`CMS_res_met` (DY and W)
  exists in 2018 only since the MET fix (`constants.RECOIL_ERAS`, from `smrun2_tau_id_v2` on); the switch `recoil_eras`
  books it where the production has it (`smrun2_tau_id_v1`: every era). `CMS_htt_ttbarShape` is the difference with (the nominal)
  against without the top-pT weight (K9): Down without it, Up the mirror 2 w - 1 around the nominal. The vsJet SF families `CMS_eff_t_dm*` equal the nominal, as in the predecessor.
- **CMSSW**: MorphingTauID2017 exists only with jvoss's local changes plus the regrouped JES declaration and the v7
  changes (no HEM, `CMS_prefiring` in every era, the Cholesky lumi total) and the v7b change (the recoil shapes in 2018
  only), versioned in `patches/`; it needs an own
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
| the scores of an installed NN payload in a copy of stored skims, as the NN friend writes them (dress rehearsal without a friend production) | `scripts/pseudo_nn_friend.py` |
| the NN input validation: the inputs and the models that read them, their binning, the b-tag WPs of NV2, the occupancy cells, the pseudo-shifts | `nn_validation.py`, `scripts/nn_validation_*.py`, `scripts/fastmtt_refit.py` |
| b-tag weight variations, embedding and fake-factor shifts, lnN table; the stages S0+ and S1 of the NN phase (ttbar shape, heavy flavour, signal theory and its inclusive normalisation) | `systematics.py`, `scripts/signal_theory_sums.py` |
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

## NN training data (ML export)

The NN phase (`docs/nn_limit_spec.md` of the workspace, 3.1, X1-X11) trains on the events the analysis selects, exported
by `shapesmith ml-export` per era (format: core README, "ML export"): one row per event of the nominal (signal) region,
with its identifiers, class, signed weight and variables. Switch `ml_labels` turns the export on (`ml.py`):

| Profile | Run YAML | Classes (in this order = class index = `predicted_class`) |
|---|---|---|
| `thesis6` (`jet_fakes: mc`, R3-R5) | `smrun2_binned_v7b_noemb.yaml` (the `tau_control` skims), `ml/thesis6/<era>` | `HH2B2Tau`; `DY` = ZTT, ZL, ZJ; `ST`; `TT`; `VV` (each with its J part); `Other` = W, TTV, EWK, single Higgs |
| `v7b5` (`jet_fakes: ff`, R6 on, N3) | `smrun2_binned_v7b_ff_nn.yaml` (re-skim `tau_ff_nn`), `ml/v7b5/<era>` | `HH2B2Tau`; `tautau` = EMB in et and mt, ZTT, TTT, STT, VVT, TTVT, EWKT in tt; `Top` = TTL, STL; `jetFakes`; `Other` = ZL, VVL, TTVL, EWKL, single Higgs |

- **jetFakes** (`v7b5`): the data of the anti-isolated region times the fake factor, as the class of the training
  (N6), and the processes the estimate subtracts (EMB or the genuine MC and the lepton fakes) in that region as
  subtraction rows (`is_subtraction`, negated weights, class `jetFakes`): with them the class is the datacard
  template (10-14 % smaller), for the proxy limit and the check of N6.
- **No QCD in `thesis6`**: the ABCD QCD of the `mc` mode is an estimate of data histograms, not of events, and is not
  exported (as in the thesis); a proxy limit of R3-R5 lacks it.
- **Signal**: the powheg κλ = 1 sample; for R3 the node_SM sample of the thesis, `-s switches.signal_sample=node_sm`
  with its own `ml_dir` (the run YAML header has the command), the powheg sample is then left out of the run.
- **Variables** (`ml_variables`, sets of `ml.py`): `thesis` (the 20 inputs of the thesis with the v15 names `n_bjets`,
  `n_jets`; `mtt_coll_approx` does not exist in v15 and is computed in the model from the legs and the MET),
  `objects` (pt, eta, phi, mass of both legs, `iso_1`, `tau_decaymode_1/2`, the b pair with `bpair_btag_value_*`, the
  two leading jets, `met`, `metphi`, `metSumEt`, FastMTT), `kinematics` (`mt_1`, `mt_2`, `mt_tot`, `pzetamissvis`,
  `mass_tautaubb`, `pt_tautaubb`, `deltaR_ditaupair`, `bpair_deltaR`), `reskim` (`bpair_mass_1/2`, `jtag_value_1/2`)
  and `truth` (`bpair_hadronFlavour_1/2`, `jet_hadronFlavour_1/2`, MC only: NaN in data and embedding rows). The
  skims of the run hold every export column; `reskim` and `truth` are in the v7b n-tuples but not in the skims before
  the re-skim of X8, so the `v7b5` run has its own skim directory and `thesis6` uses the three other sets. Values are
  those of the n-tuple, with the CROWN default -10 for an absent object (the second jet, the b pair). `iso_1` is the
  light-lepton isolation in et and mt but the raw DeepTau vsJet score of the leading tau in tt, which defines the
  anti-isolated region: never an NN input in tt; `tau_decaymode_1` is that of a tau only in tt.
- **Repeated events**: none in data and embedding rows (`event_key`, "Sample lists"; `<ml_dir>/repeated_events.json`).
  The export `ml/v7b5` of 2026-10-25 predates it: its mt 2018 file repeats 52 rows of the mutau 2018D embedding (36 of
  the class tautau, 16 subtraction rows of jetFakes). `ml/v7b5_dedupe` (2026-10-26, the same skims) is that export
  without them, otherwise equal but for `row` and the `thesis_training` split of the later mt 2018 EMB rows (241 rows);
  the M0 trainings read `ml/v7b5`, the proxy of the bbtautau-nn ladder `ml/v7b5_dedupe`.
- **Folds**: none in the files. The trainer takes `event % n` on uint64 (n = 2 or 4, N28; the model of residue r
  scores the events with `event % n == r`); `thesis_training` is the split of the thesis within a parity fold.
- **Run 2**: `shapesmith ml-merge` puts the four eras into one file per channel (column `era`); the weights keep the
  luminosity of their era, the class weights are the trainer's.

```bash
F=configs/smrun2_binned_v7b_ff_nn.yaml        # v7b5; the re-skim of X8 (new skim_dir), then the export of every era
for E in 2016preVFP 2016postVFP 2017 2018; do
  shapesmith skim -c $F -s era=$E && shapesmith ml-export -c $F -s era=$E; done
shapesmith ml-merge -c $F --era 2016preVFP --era 2016postVFP --era 2017 --era 2018 -o /ceph/sdaigler/shapesmith/smrun2_binned_v7b/ml/v7b5/run2
N=configs/smrun2_binned_v7b_noemb.yaml        # thesis6 on the stored tau_control skims (2018 for R3-R5)
shapesmith ml-export -c $N -s era=2018
shapesmith ml-export -c $N -s era=2018 -s switches.signal_sample=node_sm -s "ml_dir=/ceph/sdaigler/shapesmith/smrun2_binned_v7b/ml/thesis6_node_sm/{era}"   # R3
```

The runs with the NN categories, their cards and limits: "NN categories, limits and blinding" below.

## NN input validation (NV1-NV3)

The validation of the NN inputs before the first v7b training (`docs/nn_limit_spec.md` of the workspace, 7.1, E15, N13;
`nn_validation.py`, the core command `shapesmith input-gof`). Inputs: the union of the columns the planned models read
(`nn_validation.NN_INPUTS` with the models: the thesis recipe R3-R5 and M0 the 19 thesis columns and the legs and MET of
`mtt_coll_approx`, 28 columns; M1 the same; M2/M3 the token fields of the eight slots and the HL features), plus
`mtt_coll_approx` as the expression of the old CROWN formula. Not tested: the generator truth, the raw vsJet scores
(`iso_1` in tt) and constants (`mass_1` in mt, `tau_decaymode_1` in et and mt).

| Run YAML | Selection | Used for |
|---|---|---|
| `smrun2_binned_v7b_ff_nn_validation.yaml` | the signal region of the FF run (EMB in et/mt, jetFakes of `ff_v7b`, S0 systematics) | NV1 and NV3 in the SR, NV2 in the SR, D6 |
| `smrun2_binned_v7b_nn_validation_control.yaml` | the emb control run with the control regions, `ff_transfer` (anti_iso x FF of `ff_v7b`) | NV1 in btag0_os_pass, w_highmt_pass and lepton_antiiso, the MC jet fakes AR/SR, D6; with `-s switches.embedding=false` the EMB width of NV3 |
| `smrun2_binned_v7b_dilepton_nn_validation.yaml` | em, mm, ee with >= 2 jets | NV2 (em_tt btag1/btag2, mm/ee), NV3 in em and mm, D6 |

Each has its own re-skim under `/ceph/sdaigler/shapesmith_test/nn-validate/` (the pseudo-shifts move events across the
cuts; the control run reads the columns of the re-skim X8 and the FF friend). The pseudo-shifts (`nn_validation.py`
docstring) recompute m_vis, pt_vis, mt_1, mt_2, mt_tot, pzetamissvis, mass_tautaubb, pt_tautaubb, the dijet masses and
pT and `mtt_coll_approx` from the scaled four momenta and the propagated MET (checked against four-vectors in
`tests/test_nn_validation.py`); FastMTT is not refitted but follows the mean response of a refit (`FASTMTT_*`). The three
YAMLs set `combine.minimizer` to a fallback (`--cminFallbackAlgo Minuit2,Migrad,0:0.1`): with combine's default a failed
observed fit gives a saturated statistic of 10^2-10^3 on an unremarkable card. btag0_os_pass and w_highmt_pass veto b
tags, so the b pair and what is built from it are tested in the signal region and lepton_antiiso (`input-gof` writes no
one-bin cards; lepton_antiiso has too few et events for one).

```bash
F=configs/smrun2_binned_v7b_ff_nn_validation.yaml; C=configs/smrun2_binned_v7b_nn_validation_control.yaml; D=configs/smrun2_binned_v7b_dilepton_nn_validation.yaml
ERAS="--era 2016preVFP --era 2016postVFP --era 2017 --era 2018"; OUT=/ceph/sdaigler/nn/validation/<date>
for E in 2016preVFP 2016postVFP 2017 2018; do S="-s era=$E"
  shapesmith skim -c $F $S && shapesmith hist -c $F $S --control && shapesmith estimate -c $F $S --control
  shapesmith skim -c $C $S
  shapesmith hist -c $C $S --control --channels et,mt --regions nominal,anti_iso,btag0_os_pass,btag0_ss_pass,btag0_os_fail,btag0_ss_fail,w_highmt_pass,lepton_antiiso,lepton_antiiso_ss
  shapesmith hist -c $C $S --control --channels tt --regions nominal,anti_iso,btag0_os_pass,btag0_ss_pass,btag0_os_fail,btag0_ss_fail
  shapesmith estimate -c $C $S --control
  shapesmith skim -c $D $S && shapesmith hist -c $D $S --control --regions nominal,btag0,btag1,btag2
done
# NV1 and NV3 in the SR and the control regions: with the pseudo-shifts, and without (--exclude CMS_pseudo, own output);
# the better of the two minimizations of the autoMCStats parameters per card, 20 000 toys for every card below p = 1e-3
GOF="--toys 300 --alt-minimizer '--X-rtd MINIMIZER_no_analytic' --tail-toys 20000"
eval shapesmith input-gof -c $F $ERAS $GOF -o $OUT/gof/with_pseudo_shifts
eval shapesmith input-gof -c $C $ERAS $GOF --channels et,mt --region w_highmt_pass -o $OUT/gof/with_pseudo_shifts   # also btag0_os_pass (et,mt,tt), lepton_antiiso
python scripts/nn_validation_btag.py --dilepton $D --sr $F $ERAS -o $OUT/nv2                       # NV2
python scripts/nn_validation_fastmtt_dilepton.py -c $D $ERAS --crown <CROWN> --build <dir> -o $OUT/nv3_dilepton   # NV3 em, mm
python scripts/nn_validation_emb_width.py --emb $C --noemb-output-dir '<noemb output_dir>' $ERAS -o $OUT/nv3_emb   # NV3 EMB width
python scripts/nn_validation_transfer.py -c $C $ERAS -o $OUT/transfer                               # MC jet fakes AR/SR
python scripts/nn_validation_occupancy.py --sample sr:$F:et,mt,tt:nominal --sample btag0:$C:et,mt,tt:btag0_os_pass --sample dilepton:$D:em,mm:nominal $ERAS -o $OUT/occupancy   # D6
```

`scripts/fastmtt_refit.py` builds the FastMTT of a CROWN checkout (src/SVFit) with ROOT and runs it on numpy arrays; it
reproduces the FastMTT friend bitwise and gives FastMTT where no friend exists (em, mm) and under the pseudo-shifts.

## NN output validation (NV4, NV4b)

The validation of the outputs of a trained model (`docs/nn_limit_spec.md` of the workspace, 7.1; `nn_outputs.py`, switch
`nn_outputs`): data against the prediction for every score and category with the GoF rule of NV1 (`shapesmith
input-gof`), the fake-factor closure per NN category on the MC jet fakes and the fake-factor fractions per category.
Control variables (the scores in the 1000 fine bins of the categories): `NN_cat_<category>` (the score in each NN
category of `nn_categories`, the HH categories in the nominal region and the region of its jet fakes only below the
blinding edge of `nn_blinding`), `NN_P_<class>` (each class score over every event, without the events `NN_score` leaves
out), `NN_n_jets_<category>` and `NN_mass_2_<category>` (the NV1 decisions of 2026-10-27), `NN_ar_<HH category>` (the
HH score of anti-isolated taus only: never an event of the signal region), with `ff_transfer` the MC-only copies
`NNmc_cat_<HH category>`, `NNmc_P_<class>`, `NNmc_<input>_<HH category>` (data and embedding below the axis).

| Run | Regions | Used for |
|---|---|---|
| the fake-factor run of the model (`smrun2_binned_v7b_ff_nn_m0_v1.yaml` with the overrides of its ladder step) and `-s switches.nn_outputs=true -s switches.ff_same_sign=true -s switches.nn_blinding=<its cards>/{era}/datacards/all/blinding.json` | nominal (data, EMB, jetFakes), same_sign (jetFakes from `same_sign_anti_iso`) | NV4: the background categories, the `lo` parts, every class score (nominal); a weak hint (same sign); with `switches.ttbar_shape=true` S0 and S0+ from one `hist` (`input-gof --exclude CMS_htt_ttbarShape` for S0) |
| `smrun2_binned_v7b_nn_m0k4_nv4_control.yaml` (M0-k4; copies of the input-validation control skims with the pseudo friend) | nominal and `anti_iso` (MC jet fakes, AR x FF), `w_highmt_pass`, `lepton_antiiso`, `abcd_anti_iso` and `anti_iso_ttbar_fraction(_up)` | NV4: the FF closure per category (`scripts/nn_outputs_transfer.py`), the weak hints high mT (b veto) and lepton anti-isolation; NV4b (`scripts/nn_outputs_fractions.py`) |

NV4b compares, per category and card bin, the ttbar fraction of the anti-isolated region in the definition of the
payload (QCD = data minus every simulated process, f = TTJ / (QCD + TTJ)) with the fraction the payload applies there
(its (mt_1, n_jets) or (m_vis, n_jets) bins averaged over the jet fakes of the bin) and turns the difference into the
change of the jetFakes template through the payload's own fraction shift (the fake factor is linear in the fractions),
against the fake-factor band of the bin. The pseudo-friend copies of the control skims pass the skim reuse check of the
run except for the decay-mode non-closure that `ff_transfer` adds (a weight of the fake factor: the rows do not depend
on it).

```bash
F=configs/smrun2_binned_v7b_ff_nn_m0_v1.yaml; C=configs/smrun2_binned_v7b_nn_m0k4_nv4_control.yaml
ERAS="--era 2016preVFP --era 2016postVFP --era 2017 --era 2018"; OUT=/ceph/sdaigler/nn/nv_m0k4
STEP="<the -s overrides of the ladder step: skim_dir, nn_prefix, nn_friend_prefixes, nn_manifest; switches.ttbar_shape=true>"
NV4="-s switches.nn_outputs=true -s switches.ff_same_sign=true -s switches.nn_blinding=<cards>/{era}/datacards/all/blinding.json -s output_dir=$OUT/output/tau_ff_nn/{era}"
for E in 2016preVFP 2016postVFP 2017 2018; do
  shapesmith hist -c $F -s era=$E $STEP $NV4 --channels et,mt,tt --control --regions nominal,anti_iso,same_sign,same_sign_anti_iso
  shapesmith estimate -c $F -s era=$E $STEP $NV4 --channels et,mt,tt --control
  python scripts/pseudo_nn_friend.py -c $C -s era=$E --channel <ch> --source /ceph/sdaigler/shapesmith_test/nn-validate/skims/tau_control_nn/$E --crown <CROWN> --payload-root <payload> --model nn_classifier/<model>=nn_m0k4_
  shapesmith hist -c $C -s era=$E --control --channels et,mt --regions nominal,anti_iso,abcd_anti_iso,w_highmt_pass,lepton_antiiso,lepton_antiiso_ss,anti_iso_ttbar_fraction,anti_iso_ttbar_fraction_up
  shapesmith hist -c $C -s era=$E --control --channels tt --regions nominal,anti_iso,abcd_anti_iso,anti_iso_ttbar_fraction,anti_iso_ttbar_fraction_up
  shapesmith estimate -c $C -s era=$E --control
done
GOF="--toys 300 --alt-minimizer '--X-rtd MINIMIZER_no_analytic' --tail-toys 20000"
eval shapesmith input-gof -c $F $ERAS $STEP $NV4 --channels et,mt,tt $GOF --variables <NN_cat_*,NN_P_*> -o $OUT/nv4/gof_s0plus
eval shapesmith input-gof -c $F $ERAS $STEP $NV4 --channels et,mt,tt $GOF --variables <NN_cat_*,NN_P_*> --exclude CMS_htt_ttbarShape -o $OUT/nv4/gof_s0
eval shapesmith input-gof -c $F $ERAS $STEP $NV4 --channels et,mt,tt $GOF --region same_sign -o $OUT/nv4/weak_s0
eval shapesmith input-gof -c $C $ERAS --channels et,mt $GOF --region w_highmt_pass -o $OUT/nv4/weak_s0   # also lepton_antiiso
python scripts/nn_outputs_transfer.py -c $C $ERAS -o $OUT/nv4/transfer                        # FF closure per NN category
python scripts/nn_outputs_fractions.py -c $C $ERAS --limit-run '<the ladder step>/shapesmith/<variant>/{era}' -o $OUT/nv4b   # NV4b
```

NV4b is work in progress: the template-response estimate uses only the ttbar-fraction Up displacement.
In saturated payload cells this displacement can be zero, so the corresponding bins have no verdict
(`within_ff: null`, shown as `-`) and are excluded from the assessed-bin count. A report with such bins
does not establish NV4b closure; an informative fraction response still has to be evaluated.

## NN categories, limits and blinding

The chain of the NN phase from the training data to the limit (`docs/nn_limit_spec.md` of the workspace, 3-8; the
decisions N1-N28 of 2026-10-25): the categories, their binning and blinding (SH1-SH6), the statistics commands
(SH9-SH12), the systematics stages and run YAMLs (SH7, SH8, SH13-SH15). The core README describes the mechanics
("Score categories", "Limits and statistical checks"); this section names the analysis's choices.

1. **Training data**: `ml-export` (above). The trainer (repository `bbtautau-nn`, N2) writes per model and channel one
   ONNX file per outer-fold residue and a manifest with the inputs and classes in order and the fold count (`event % n`,
   n = 2 or 4, N28: the model of residue r scores the events with `event % n == r` and never saw them); its `install`
   puts them into the CROWN payload `payloads/nn_classifier/<model>/<era>/<scope>/manifest.json`.
2. **NN friend**: CROWN bbtautau `sm_nn_classifier` evaluates every model of a production (its `MODELS`: payload
   directory and column prefix) and writes `<prefix>predicted_class` (the class index in the order of the manifest),
   `<prefix>predicted_max_value` and `<prefix>nn_score_<class>`, -1 outside its gate (the tau skim selection with
   control regions, without the charge requirement, N21). It runs on the full list of every tau run, chained on the
   FastMTT friend (KingMaker `sm_nn_classifier: {friend_tag: <tag>, requires: [sm_fastmtt]}`), the data under
   `smrun2_binned_v7` and the MC and embedding under `smrun2_binned_v7b`, `--shifts None` until P6. Productions
   (spec 9): FR1 `nn_thesis2018_v1` (2018, the thesis recipe trained on node_SM and on powheg, 6 classes), FR2 `nn_m0_v1`
   (Run 2, M0 and M0-k4, 5 classes), FR3 `nn_step2_v1` (step 2); each a fresh tag.
3. **NN runs** (SH13): each is the run it copies with the friend entries after `fastmtt`, `nn_friend: true`,
   `fit_categories: nn`, the NN switches below and its own `skim_dir` (a new friend needs new skims; the production
   skims stay), without export and gallery (`tests/test_nn_runs.py`). The model of a run is one column prefix of its
   friend (`nn_prefix`); `nn_friend_prefixes` lists every model of the friend, whose columns the skims keep, so the run of
   the other model reuses the skims with `-s switches.nn_prefix=...` and its own `output_dir` (the YAML headers have the
   commands). The prefixes are those of the friend's `MODELS` (CROWN `sm_nn_classifier.py`, empty until FR1/FR2 exist).

   | Run YAML | Friend | Runs | NN switches | Background model |
   |---|---|---|---|---|
   | `smrun2_binned_v7b_nn_thesis2018_v1.yaml` | FR1, 2018 | R4 (powheg); R3 with `-s switches.signal_sample=node_sm -s switches.nn_prefix=nn_thesis_nodesm_` and its own `output_dir` (skim once more with `signal_sample=node_sm` for the node_SM sample) | `nn_prefix: nn_thesis_powheg_`, `nn_friend_prefixes: [nn_thesis_nodesm_, nn_thesis_powheg_]`, `nn_categories: thesis`, `nn_binning: thesis` (classes thesis6 of the `mc` mode) | `noemb`: MC jet fakes, ABCD QCD, no embedding; et, mt, tt (the final state `all` without em, as the thesis) |
   | `smrun2_binned_v7b_ff_nn_thesis2018_v1.yaml` | FR1, 2018 | R5 (the R4 model) | as R4 and `nn_classes: [HH2B2Tau, DY, ST, TT, VV, Other]` (thesis6: the `ff` mode would give v7b5) | the final one: EMB in et/mt, `jetFakes` from `ff_v7b`, em; `datacards --channels et,mt,tt` for `all` without em |
   | `smrun2_binned_v7b_ff_nn_m0_v1.yaml` | FR2, every era | R6, R7 (M0), the baseline; R8 (M0-k4) with `-s switches.nn_prefix=nn_m0k4_` and its own `output_dir` | `nn_prefix: nn_m0_`, `nn_friend_prefixes: [nn_m0_, nn_m0k4_]`, `nn_categories: k1` (`k2` if R7 chooses it, N10), `nn_binning: data_blind` (classes v7b5) | the final one with `ff_v7b` |
   | `smrun2_binned_v7b_ff_w_nn_m0_v1.yaml` | FR2, every era | the same with W+jets fake factors (N7: trained with `ff`, limits for `ff` and `ff_w`) | as above | the final one with `ff_v7b_w` |

   `nn_manifest` (unset in the YAMLs until the payloads exist) checks the classes against the installed manifest.

   ```bash
   F=configs/smrun2_binned_v7b_ff_nn_m0_v1.yaml
   for E in 2016preVFP 2016postVFP 2017 2018; do S="-s era=$E"
     shapesmith skim -c $F $S -s switches.heavy_flavour=true   # once; with the S0+ columns, so the skims serve S0 and S0+
     shapesmith hist -c $F $S && shapesmith estimate -c $F $S && shapesmith datacards -c $F $S && shapesmith fit -c $F $S
     shapesmith datacards -c $F $S --variant stat_only --no-systematics && shapesmith fit -c $F $S --variant stat_only; done
   shapesmith combine -c $F --era 2016preVFP --era 2016postVFP --era 2017 --era 2018 -o /ceph/sdaigler/shapesmith/smrun2_binned_v7b/combination/ff_nn_m0_v1
   ```
   `hist` runs **without** `--skip-systematics` (which books no variation at all, so the cards would have no shapes)
   and fills every NN category in 1000 fine bins plus the unweighted counts `NN_score%raw` the binning needs: shapes of
   an earlier ShapeSmith version must be filled again.
4. **Categories** (`nn_categories`, spec 6.2, N10-N12): K1 (`k1`) is the argmax over the classes per channel and era,
   one category per class named after it (`HH2B2Tau`, `tautau`, `Top`, `jetFakes`, `Other` for v7b5) with the cut
   `(<prefix>predicted_class == i) & (n_bjets < 3)`; the discriminant is the score of the predicted class
   (`<prefix>predicted_max_value`, in the HH category P(HH)), finely binned; the background categories have at most 5
   bins and act as normalisation controls. K2 (`k2`) splits the HH category into `HH2B2Tau_btag1` (`n_bjets == 1`) and
   `HH2B2Tau_btag2` (`== 2`; about 40 % of the signal has one b jet); K2 replaces K1 if it gains ≥ 3 % and > 2σ_paired
   in R7, then scheme and binning algorithm are fixed. `thesis` is the plain argmax of the thesis (R3-R5). Events with
   `n_bjets ≥ 3` are cut from K1 and K2 (N12, -2.3 % signal), and with NN categories em_tt is split into
   `em_tt_btag1` and `em_tt_btag2` without btag3p (SH4). Card bins of a K1 tau channel: `htt_<ch>_1lo_<era>`,
   `htt_<ch>_1hi_<era>`, then `htt_<ch>_2_<era>` to `htt_<ch>_5_<era>`; the background-only fit summaries name them
   `HH2B2Tau lo`, `HH2B2Tau hi`, `tautau`, ...
5. **Binning** (`nn_binning`, spec 6.3, the same data-blind algorithm for every model from R6 on): `datacards` resolves
   the edges after `estimate` from the nominal templates (the rules need the jetFakes template; data are never read).
   `data_blind` (rule set C', decided 2026-10-26): the HH category starts from 50 bins of equal signal on P(HH) in
   [1/K, 1] and merges from the top until every bin has a total background ≥ 1.0 weighted events with a relative
   statistical uncertainty ≤ 50 %, ≥ 1 unweighted event of each raw-count group whose share of the category's
   background (nominal templates, P ≥ 1/K) is ≥ 10 % (in tt Top, 3-4 %, is exempt), and no template (signal, every
   background, jetFakes) below -1 % of the bin's total background; the remaining low bins join the lowest accepted
   bin, at most 15 bins. The background categories start from 5 bins of equal background, with the same raw-count and
   negative rules, ≤ 10 % and ≤ 5 bins. The rules before C' (≤ 30 %, every raw-count group, every template ≥ 0) cost
   ×1.27 on the limit of R6_wp, almost all in tt. The raw-count groups follow the background model
   (`ml.raw_count_groups`): Top (TTL, STL), tautau (EMB or the genuine MC di-tau) and jetFakes (its anti-isolated
   data) with fake factors; TT, DY and the ABCD QCD in `mc` mode.
   jetFakes (QCD in `mc` mode) is clipped at 0 after the rebinning (SH6, listed under `clipped` in
   `template_changes.json`). `thesis`: the custom edges of the thesis (HH) and 20 equal bins (others), merged by the
   AutoRebin(1.0, 0.9) of the thesis, for R3-R5 and as the cross-check of R6 and R7. The edges per channel and category
   go to `<card dir>/nn_binning.json` (with the rules, the share of every raw-count group and the exempt groups);
   datacards warns when a blinded category has fewer than 3 bins (N11: the 2016
   halves stay separate cards).
6. **Blinding** (spec 6.4, N14), see the rules below.
7. **Statistics** (spec 7.2, commands of the core README, all with `--variant KEY`): `fit` gives the prefit Asimov
   limits (`-t -1 --expectSignal 0`, converged with `combine.precise`) on μ and σ(ggF HH) for the model comparisons;
   `fit --background-only` the masked background-only fit with pulls and the prefit/postfit stacks of every card bin;
   `limit-aposteriori` the expected limit of the result (the masked background-only fit to the data, its snapshot, then
   `AsymptoticLimits -t -1 --toysFrequentist --bypassFrequentistFit`); `injection` toys at μ = 0, 1 and the expected
   limit (linearity, unbiased pulls); `limit-hybridnew` the HybridNew cross-check of the final expected limit; `gof` the
   saturated GoF with toys per channel, era and combined; `impacts` (data or `--asimov`) with pull plots; `limit-overview`
   the table and Brazil plot per channel, era and combined. `datacards --variant stat_only --no-systematics` puts the
   "statistics only" cards beside S0; the r range, minimizer, timeout and parallel jobs are `combine:` options of the run
   YAML. The NN run YAMLs set the fallback of the validation runs, `--cminFallbackAlgo Minuit2,Migrad,0:0.1`: combine's
   default reports a failed fit to the data as it is; a fit that converges (the Asimov limits) stays the same. `fit
   --limits-only` (and `combine --limits-only`) leaves out the significance and the best fit.
8. **Units** (N23): μ = σ/σ_SM of ggF HH → bbττ first; σ(ggF HH) = r × 30.40 fb (`Analysis.signal_cross_section`), from
   a second limit with the cross-section theory nuisances of the signal (`QCDscale_HH`, `PDF_alphas_HH`) frozen and its
   acceptance, shape and BR uncertainties kept; ggF only, named so. Every result is reported with "statistics only"
   (data statistics and autoMCStats) next to its stage.

**Dress rehearsal without a friend production** (`scripts/pseudo_nn_friend.py`): it scores the events of stored skims
with installed payloads exactly as the friend does (CROWN's `nn_payloads.read_payload` and gate, the model of residue
`event % n` on uint64, one event per call, -1 outside the gate) and writes a copy of the skims with the friend columns
of every `--model PAYLOAD=PREFIX` and the manifests extended by the friend of the run YAML, then checks the copy with
ShapeSmith's skim reuse check. A copy of a run YAML with `skim_dir` on the copy then runs `hist` and the rest unchanged.

**Blinding rules** (spec 6.4, 8):
- Every HH category is two card channels, `htt_<ch>_<k>lo_<era>` and `htt_<ch>_<k>hi_<era>` (combine can mask whole
  channels only), split at S/B(μ = 1) = 0.5 % from the nominal templates of the model, before data are looked at: the hi
  part starts at the lowest bin with S/B > 0.5 % and takes every bin above it (even one that falls back below); no lo
  channel if the lowest bin passes, one unsplit unmasked channel if none does.
- The hi channels have the Asimov background prediction as `data_obs`; the lo channels, the background categories and
  the control regions have the real data. `datacards` writes `<output_dir>/datacards[/<variant>]/<final
  state>/blinding.json` next to `combined.txt` (format: core README, "Score categories"); `masked` lists exactly the
  card bins whose `data_obs` is the Asimov prediction; a card without NN categories has no file, and a missing file
  masks nothing. `combine` merges the `masked` lists of the cards of the eras.
- Every fit to real data (`fit --background-only`, `combine --background-only`, `gof`, `impacts` on data, the fit of
  `limit-aposteriori`, the toys of `injection --a-posteriori`) builds the workspace with `--channel-masks` and adds
  `mask_<bin>=1` for every bin in `masked` to its `--setParameters`; a prefit Asimov fit (`-t -1`) needs no mask. The
  summaries list the masked channels and show no data for them.
- Plots and gallery yields of the NN categories show the prediction in the hi bins (`plot --blind` hides the data of the
  whole histogram). These plots and the prefit and postfit stacks of `fit --background-only` draw the bins of the card
  with equal widths and the bin edges as tick labels, in events per bin (`plot --normalize-by-bin-width`: dN/dx on the
  score axis instead).
- Only the signal category is blinded (multiclass argmax categories, decided 2026-10-25): the control variable
  `NN_score` (the score of the predicted class over every event; `hist --control`, `plot --control`, the gallery)
  leaves out the events predicted as `HH2B2Tau` in the nominal region and in the anti-isolated region whose events are
  its jet-fake estimate (`anti_iso`, `abcd_anti_iso` in `mc` mode), so that its prediction holds the same events as its
  data (`variables.nn_score_blinding`: their score minus 2, below the axis). Every other region keeps every event (same
  sign, lepton anti-isolation, btag0, high mT); the os pass and fail regions of btag1, btag2 and btag3p
  (`control_regions`) lie inside the nominal or the anti-isolated region and lose those events too, while their
  same-sign QCD estimate keeps them.
- Not blinded, and never read by combine: `shapes.root` and its index `shapes.json` (the sum of every histogram, the
  data of a whole NN category included) and `sync` (the fit categories in their 1000 fine bins with the full data).
  Their nominal-region data of the NN categories stay unread until unblinding.
- Unblinding (S2: real data in hi, no masks; `--unblind` of the fit commands, logged loudly) only after NV4, NV5 and
  the checks of spec 7.2 have passed with S1, the impacts are reviewed, and with the release of the analysis owner; no
  run YAML or default unblinds.

**Systematics stages** (spec 8; `systematics.py`, switches of the tau run, all off by default so that every run YAML
stays at S0; `tests/test_nn_systematics.py`):

| Stage | Nuisances | Switches |
|---|---|---|
| S0 (now, nominal only, E34) | data statistics and autoMCStats; the lnN (Cholesky lumi, cross sections, HH theory, BR, lepton and τ ID, EMB ES, syst_TES as acceptance lnN); the weight shapes of the nominal trees (b tag, prefiring as lnN, FF friend shifts with the FF DM non-closure, EMB ID weight factors) | the v7b FF runs as they are (`study_uncertainties`, `symmetrise_ff_shifts`) |
| S0+ (ShapeSmith only, before P6) | + the ttbar top-pT shape (SH7), + the heavy-flavour normalisation (SH8; a no-op in K1 and K2, see below), + the pseudo-shifts of NV9 as preliminary shapes of the NN outputs (see below); em_tt btag1/btag2 (SH4) and the merge and clip of negative jetFakes bins (SH6) come with the NN categories at every stage | `ttbar_shape: true`, `heavy_flavour: true`, `pseudo_shifts: true` |
| S1 (after P6) | + every kinematic CROWN shift (needs the NN friend with `--shifts all` and the `fastmtt` friend with shifts), PS, PU, the signal theory shapes (SH14) | `crown_shifts: true`, `signal_theory: [qcd_scale, parton_shower, pdf]` |
| S2 | unblinding, see the blinding rules | - |

- **SH7 `CMS_htt_ttbarShape`** (v7_analyse 4.1, 11.2 no. 2): the ttbar n_jets slope (≥ 4 against 2 jets -3 to -7 %
  in em, to -14 % in em btag1) has no shape uncertainty in the cards. As in the tau-ID measurement (K9, TOP: with
  against without the weight): Down without `topPtReweightWeight`, Up its mirror 2 w - 1 around the nominal, on every
  process carrying the top-pT weight (`weights.TOP_PT`: TTT, TTL, TTJ of the tau channels, TT of em), one nuisance for
  every channel and era (theory, K), in the nominal region. The weight lies in [0.83, 1.06] in the v7b skims; in 2018 it
  moves ttbar by ±0.5-0.8 % at 2 jets and ±2.2-2.5 % at ≥ 4 jets (TTL of mt and et), ±3.4-4.7 % for TTT in tt, and the
  simulated backgrounds by ±1.2 / 0.9 / 1.7 % (et / mt / tt), so it covers about a third to a half of the slope; the
  ISR/FSR PS weights of ttbar follow with P6.
- **SH8 `CMS_htt_heavy_flavour`** (v7_analyse 4.5, 11.2 no. 1; N12): the simulation lacks heavy flavour at ≥ 3 b tags
  (data/prediction of btag3p 1.24-1.37 in em, 1.26-1.35 in mm + ee with the DY table, 0.99-1.29 in et and mt, in every
  era; the b-tag SF +1σ reaches only 1.25-1.35): ttbb, Z/V + bb and g → bb. Its component is the events with
  `n_bjets ≥ 3` whose b pair is two heavy-flavour jets (`bpair_hadronFlavour_{1,2}` 4 or 5), on every simulated
  background (not the signal, EMB or the data-driven jetFakes/QCD, whose data contain the heavy flavour), in the tau
  channels and em; Up × 1.30, Down × 1/1.30 (lnN 1.30, the size of the deficit), written as a weight variation so that
  its share sets its size per bin, one nuisance (theory, K). Below 3 b tags the b pair of ttbar is that of the top decays
  and the DY table corrects DY in btag1 and btag2 (`CMS_htt_dy_table`), so the component starts at 3 b tags. In 2018 it
  is 79-85 % of the btag3p backgrounds and 0.9-1.6 % of all simulated backgrounds (et, mt, tt): +30 % on it gives
  +24-26 % on btag3p and +0.5 / 0.5 / 0.3 % on the simulated backgrounds (et / mt / tt). With NN categories no category has `n_bjets ≥ 3` (K1/K2 cut btag3p, em_tt
  is split into btag1 and btag2), so the nuisance is a no-op there and `datacards` leaves it out (logged); it acts on the
  control categories and is the prerequisite for btag3p in the NN categories. The ttbb fraction of ttbar itself
  needs the ttbb identifier of P6 (N19). It reads columns of the re-skim X8 (`skims/tau_ff_nn` and the NN runs): skim
  with the switch on, or the skim reports the missing columns.
- **NV9 pseudo-shifts of the NN outputs** (S0+; `pseudo_shifts` with the NN friend): the pseudo-shifts of the input
  validation (`nn_validation.py`: jets ±3 %, τh ±1.5 %, the MET propagated, the derived inputs and the cuts recomputed)
  also read the outputs of the model under the shift, the friend columns `<column>__<shift>` (the model evaluated on the
  shifted inputs behind the shifted gate). Until the shift production P6 the pseudo friend writes them
  (`scripts/pseudo_nn_friend.py --pseudo-shifts`) into copies of skims that keep the events passing under a shift, the
  skims of the input validation (`/ceph/sdaigler/shapesmith_test/nn-validate/skims/tau_ff_nn`); em from the skims of the
  run. The pairs are symmetrised in the cards like those of NV1. Objects below the CROWN thresholds are not in the
  n-tuples, so an Up shift misses the events that would enter there; n_jets and n_bjets stay.
- **SH14 signal theory** (S1, N19, N23): the QCD scales `QCDscale_ren_HH_acc` and `QCDscale_fac_HH_acc` from the CROWN
  shifts `lhe_scale_weight__QCDscale_{ren,fac}{Up,Down}` (bbtautau `variations/theory.py`, muR or muF 2 and 0.5), the
  parton shower `ps_isr_HH`, `ps_fsr_HH` (ISR/FSR scale 2 Up, 0.5 Down, NanoAOD `PSWeight`) and `pdf_HH_acc` (1 ± the
  Hessian sum of `LHEPdfWeight`, CROWN `event::reweighting::LHEpdf`), each the varied weight over its inclusive mean
  (`systematics.SIGNAL_THEORY_NORM`, measured per era by `scripts/signal_theory_sums.py` on the NanoAOD of the powheg
  sample: the Runs sums `LHEScaleSumw`, `PSSumw`, and the PDF from the Events of one file), so only acceptance and shape
  remain and nothing is counted twice with `QCDscale_HH` and `PDF_alphas_HH`. Inclusively the scales move the signal by
  -10.6/+12.0 % (muR) and -1.8/+2.3 % (muF), the PS by ≤ 0.1 %, the per-event PDF sum by ±1.8 %. The v7b n-tuples have
  only the nominal `lhe_scale_weight` (no `LHEScaleWeight`, `PSWeight` or `LHEPdfWeight`, checked on the HH files of
  2018): the QCD scales need the CROWN shift production (P6); the PS and PDF columns need producers in `sm_config` (their
  names `ps_weight__ps_{isr,fsr}*` and `lhe_pdf_weight__pdf*` are aligned with them at P6). Without the columns the skim
  reports them missing.

## Prerequisites outside this repository

- NN friend trees from a trained model and the CROWN friend `sm_nn_classifier` (NN spec 3.4; CROWN bbtautau
  `feat-new-sm-bbtautau-config`, see "NN categories, limits and blinding"). `<prefix>predicted_class` indexes the classes
  of the model (`nn_classes`, else the label profile: `ml.ML_CLASSES`, the `classes` of the export's `classes.json` and
  of the payload's `manifest.json`). With CROWN shifts the NN friend must be produced with `--shifts all`, or the NN
  category and score stay nominal under them.
- Fake-factor friends (`fake_factor`, `fake_factor_1/2`, the raw `fake_factor_raw`, `fake_factor_1/2_raw` and their
  shifts) from CROWN `fake_factors_friend_config.py`, on data, MC and embedding. A new FF payload gets a new friend tag
  (a new `ntuples.friends` base); the skim contract records the friend bases, so the stored skims then need `skim --force`.
- Embedding ntuples under the same `ntuples.base` and production tag as the data/MC they are combined with, produced
  with `--scopes et,mt,tt --shifts all`.

Until they exist: `nn_friend: false`, `jet_fakes: mc`, `embedding: false`.
