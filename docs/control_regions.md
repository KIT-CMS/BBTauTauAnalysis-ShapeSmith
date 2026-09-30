# Control regions

The two Analyses use the same `sm2018_binned_v4` CROWN ntuples and separate output directories:

| Final states | Channels | Run configuration |
|---|---|---|
| At least one hadronic tau | `et` = eτh, `mt` = μτh, `tt` = τhτh | [sm2018_binned_v4.yaml](../configs/sm2018_binned_v4.yaml) |
| Two light leptons | `em` = eμ, `mm` = μμ, `ee` = ee | [sm2018_binned_v4_dilepton.yaml](../configs/sm2018_binned_v4_dilepton.yaml) |

OS means opposite-sign (`q_1 * q_2 < 0`), SS means same-sign (`q_1 * q_2 > 0`).
The selections below follow [cuts.py](../src/bbtautau_shapesmith/cuts.py) and
[dilepton.py](../src/bbtautau_shapesmith/dilepton.py).

## What does `control_regions: true` mean?

It is a boolean under `switches` in the **tau** run configuration. It is already enabled in
`configs/sm2018_binned_v4.yaml`; the default in [switches.py](../src/bbtautau_shapesmith/switches.py)
is `false` if the switch is omitted. The relevant entries are:

```yaml
switches:
  jet_fakes: mc
  control_regions: true
```

The switch adds the diagnostic regions listed below: **20 per et/mt channel and 16 in tt**.
It also widens the ShapeSmith Parquet skim so their events survive:

| Skim requirement | `control_regions: false` | `control_regions: true` |
|---|---|---|
| Charge | Both OS and SS | Both OS and SS |
| Tau vsJet ID | VVVLoose on every tau leg | VVVLoose on every tau leg |
| Jets / b-tags | `n_bjets >= 1` and `bpair_pt_2 > 0` | `n_jets >= 2` **or** the previous requirement |
| Lepton isolation (`et`, `mt`) | `iso_1 < 0.15` | `iso_1 < 0.5` |
| Transverse mass (`et`, `mt`) | `mt_1 < 70` GeV | No mT cut |

Both charges and VVVLoose taus are already retained with `false` for the background-estimation
regions. The nominal event selection and its estimator are unchanged by the switch. The OR in
the jet requirement preserves nominal events because `n_jets` and `n_bjets` can use different
pT thresholds. Triggers, tau anti-electron/muon requirements and lepton vetoes remain in the skim.

The switch requires `jet_fakes: mc`; combining it with `jet_fakes: ff` is rejected. The dilepton
Analysis always defines its control regions and accepts only `sample_lists` under `switches`;
do not add `control_regions` to its YAML.

The YAML switch and the command-line options have different jobs:

| Setting / option | Effect |
|---|---|
| `switches.control_regions: true` | Defines additional tau regions and widens the tau skim |
| `hist --control` | Fills control variables in the inclusive category into `control_shapes.root` |
| `hist --regions all` | Fills `nominal` and every region defined for each selected channel |
| `hist --regions NAME1,NAME2` | Fills only the specified regions |
| `plot --control --region NAME` | Plots control-variable histograms from one already-filled region |

`--control` alone does **not** request all regions. Without `--regions`, `hist` fills nominal
plus the estimator's regions for nonsignal processes (only nominal for signal); the dilepton
Analysis has no estimator and therefore defaults to nominal only.

The v4 CROWN production already retains the events needed here; no separate CROWN production
is needed. If the tau Parquet skim was made with `control_regions: false`, rebuild it using
`shapesmith skim -c configs/sm2018_binned_v4.yaml --force`. Changing the YAML cannot recover
events discarded by an earlier skim. `hist` always fills the requested regions and keeps the other
histograms of an existing `control_shapes.root`.

## Channels with hadronic taus: `et`, `mt`, `tt`

### Nominal and background-estimation regions

The nominal baseline requires OS, at least one b-tag and `bpair_pt_2 > 0`, Medium vsJet ID
on every tau leg, tau anti-electron/muon ID, trigger requirements and extra-lepton vetoes.
In `et`/`mt`, it also requires `iso_1 < 0.15`, `mt_1 < 70` GeV and the dilepton veto;
in `tt`, both taus must have pT > 40 GeV. The `et`/`mt` trigger selections require tau
pT > 30 GeV and electron/muon pT > 33/25 GeV, respectively, plus the single-lepton trigger flag.

**Tau pass/fail** refers to the vsJet ID, not the light-lepton isolation:

| Channel | Pass | Fail |
|---|---|---|
| `et`, `mt` | Tau leg 2 passes Medium | Tau leg 2 passes VVVLoose and fails Medium |
| `tt` | Both tau legs pass Medium | Both pass VVVLoose and **exactly one** fails Medium |

These regions exist with `jet_fakes: mc` even when `control_regions` is `false`.
Only the listed cuts change relative to nominal:

| Region | Charge | Tau ID | Role |
|---|---|---|---|
| `nominal` | OS | Pass | Nominal selection; ABCD target A |
| `same_sign` | SS | Pass | Same-sign comparison |
| `abcd_same_sign` | SS | Pass | ABCD input C; same selection as `same_sign` |
| `abcd_anti_iso` | OS | Fail | ABCD input B |
| `abcd_same_sign_anti_iso` | SS | Fail | ABCD input D |

In `jet_fakes: ff` mode, the named regions are instead `same_sign` and `anti_iso` (OS, tau
fail, fake-factor friend weight); the three `abcd_*` regions are absent and the additional
diagnostic regions below cannot be enabled.

### Additional b-tag / charge / tau-ID regions

With `control_regions: true`, **all three tau channels** get the following 16 regions.
Every entry requires `n_jets >= 2` and the indicated b-tag count. This replaces the entire
nominal b-tag/b-pair cut, including `bpair_pt_2 > 0`. All other baseline cuts remain;
in particular, `et`/`mt` still require `mt_1 < 70` GeV and `iso_1 < 0.15`.

| b-tag count | OS, pass | OS, fail | SS, pass | SS, fail |
|---|---|---|---|---|
| 0 | `btag0_os_pass` | `btag0_os_fail` | `btag0_ss_pass` | `btag0_ss_fail` |
| 1 | `btag1_os_pass` | `btag1_os_fail` | `btag1_ss_pass` | `btag1_ss_fail` |
| 2 | `btag2_os_pass` | `btag2_os_fail` | `btag2_ss_pass` | `btag2_ss_fail` |
| ≥3 | `btag3p_os_pass` | `btag3p_os_fail` | `btag3p_ss_pass` | `btag3p_ss_fail` |

### Additional W and lepton-isolation regions: `et`, `mt` only

These four regions also require `control_regions: true`. They are not defined in `tt`.
Unlisted baseline cuts remain unchanged.

| Region | Charge | Tau ID | Jets / b-tags | mT(lepton, MET) | Light-lepton isolation | Main use |
|---|---|---|---|---|---|---|
| `w_highmt_pass` | OS | Pass | ≥2 jets, 0 b-tags; no b-pair cut | >80 GeV | <0.15 | W+jets comparison with a passing tau |
| `w_highmt_fail` | OS | Fail | ≥2 jets, 0 b-tags; no b-pair cut | >80 GeV | <0.15 | W+jets comparison with a failing tau |
| `lepton_antiiso` | OS | Pass | Nominal: ≥1 b-tag, `bpair_pt_2 > 0` | <70 GeV | 0.15 ≤ `iso_1` < 0.5 | Light-lepton isolation sideband |
| `lepton_antiiso_ss` | SS | Pass | Nominal: ≥1 b-tag, `bpair_pt_2 > 0` | <70 GeV | 0.15 ≤ `iso_1` < 0.5 | Same-sign light-lepton isolation sideband |

The additional diagnostic regions are raw data/MC comparisons with **no data-driven QCD or
jetFakes estimate**. In MC their tau-fail selections apply the passing tau-ID scale factor only to
passing genuine-tau legs; failed genuine taus stay uncorrected. The muon anti-isolation
sidebands do not use the MC muon isolation scale factor. The embedded sample keeps its own weights
in both (its iso-binned muon SF covers the sideband).

Including nominal and the four existing same-sign/ABCD names, `--regions all` selects
**25 region names in each of et/mt and 21 in tt** for the supplied MC-fakes configuration.

```bash
# Requires a skim made with control_regions: true; an existing histogram file keeps its other histograms.
shapesmith hist -c configs/sm2018_binned_v4.yaml --control --regions all --skip-systematics
shapesmith plot -c configs/sm2018_binned_v4.yaml --control --channels et,mt --region w_highmt_pass --variables mt_1,met,pt_2,n_bjets
shapesmith plot -c configs/sm2018_binned_v4.yaml --control --channels et,mt --region w_highmt_fail --variables mt_1,met,pt_2
shapesmith plot -c configs/sm2018_binned_v4.yaml --control --region btag2_os_pass --variables m_vis,met,n_bjets
shapesmith plot -c configs/sm2018_binned_v4.yaml --control --region btag1_ss_fail --variables pt_2,m_vis
shapesmith plot -c configs/sm2018_binned_v4.yaml --control --channels et,mt --region lepton_antiiso --variables iso_1,mt_1

# For the nominal comparison, add the ABCD QCD estimate after filling its input regions.
shapesmith estimate -c configs/sm2018_binned_v4.yaml --control
shapesmith plot -c configs/sm2018_binned_v4.yaml --control --region nominal --variables yield,m_vis,met
```

## Light-dilepton channels: `em`, `mm`, `ee`

These channels have no hadronic tau and no tau pass/fail split. Both leptons have isolation
<0.15, and extra-electron/muon vetoes apply. The nominal charge is OS.

| Channel | Leptons and pT thresholds | Trigger | Mass window | Nominal jets / b-tags | Data | Main use |
|---|---|---|---|---|---|---|
| `em` | Electron (leg 1) >15 GeV, muon (leg 2) >26 GeV | Single muon | None | ≥2 jets; inclusive in b-tags | SingleMuon | ttbar kinematics and b-tag multiplicity |
| `mm` | Muons: leg 1 >26 GeV, leg 2 >15 GeV | Single muon | 70 < m(ll) < 110 GeV | Inclusive in jets and b-tags | SingleMuon | Z normalisation, recoil, Z+heavy flavour |
| `ee` | Electrons: leg 1 >34 GeV, leg 2 >15 GeV | Single electron | 70 < m(ll) < 110 GeV | Inclusive in jets and b-tags | EGamma | Electron cross-check of Z+jets |

The 26/34 GeV thresholds match the offline thresholds inside CROWN's trigger flags.
In `mm`/`ee`, leg 1 is the leading-pT lepton. CROWN writes every data stream into every scope;
[samples.py](../src/bbtautau_shapesmith/samples.py) routes SingleMuon to `em`/`mm` and EGamma to `ee`.

**Every dilepton channel has these six region names**, without a `control_regions` switch:

| Region | Charge | Jets | b-tags | Other cuts |
|---|---|---|---|---|
| `nominal` | OS | ≥2 in `em`; inclusive in `mm`/`ee` | Inclusive | Channel baseline above |
| `btag0` | OS | ≥2 | Exactly 0 | Baseline lepton, trigger, veto and mass cuts |
| `btag1` | OS | ≥2 | Exactly 1 | Baseline lepton, trigger, veto and mass cuts |
| `btag2` | OS | ≥2 | Exactly 2 | Baseline lepton, trigger, veto and mass cuts |
| `btag3p` | OS | ≥2 | ≥3 | Baseline lepton, trigger, veto and mass cuts |
| `same_sign` | SS | ≥2 in `em`; inclusive in `mm`/`ee` | Inclusive | Baseline with only the charge flipped |

The four b-tag regions partition the OS baseline events with ≥2 jets. `same_sign` provides a
qualitative check of nonprompt backgrounds; it keeps the Z mass window in `mm`/`ee`.
There are no combined `btag*_ss` or lepton anti-isolation regions in this Analysis.

MC is unsplit (no gen-match selections, no fake estimate). Both lepton legs receive ID/isolation
or reconstruction weights, the trigger scale factor belongs to the triggering leg, and TT keeps
its top-pT weight. Only the correlated/uncorrelated b-tag variations are booked; omit
`--skip-systematics` to fill them.

```bash
# Skim the dilepton channels once; their channel directories are separate from et/mt/tt.
shapesmith skim -c configs/sm2018_binned_v4_dilepton.yaml
shapesmith hist -c configs/sm2018_binned_v4_dilepton.yaml --control --regions all --skip-systematics
shapesmith plot -c configs/sm2018_binned_v4_dilepton.yaml --control --channels em --region btag2 --variables yield,pt_1,pt_2,met,n_bjets
shapesmith plot -c configs/sm2018_binned_v4_dilepton.yaml --control --channels mm,ee --region nominal --variables yield,m_vis,pt_vis,n_jets
shapesmith plot -c configs/sm2018_binned_v4_dilepton.yaml --control --channels mm,ee --region btag1 --variables yield,m_vis,pt_vis
shapesmith plot -c configs/sm2018_binned_v4_dilepton.yaml --control --region same_sign --variables yield,m_vis,met
```

## Reading the comparisons

Start with `em` `btag1`/`btag2` (yield, lepton pT, MET, b-tag multiplicity), then the inclusive
Z yields and pT(ll) in `ee`/`mm` before their b-tag bins, then the tau W high-mT pass/fail and
SS regions. Agreement in `em` with a discrepancy only in the tau channels points to tau or
fake modelling; a common b-tag trend points to tagging or flavour composition. A residual
in a region without a data-driven fake/QCD estimate is not automatically an MC defect.

The one-bin `yield` variable counts every selected event; finite-range kinematic histograms
are no substitute for it. Zero-b regions do not require a reconstructed b pair; CROWN's b-pair
quantities are sentinel values there, so use lepton, jet and MET variables.

The full region collection is not mutually exclusive: b-tag bins overlap with nominal,
and tau `same_sign` duplicates `abcd_same_sign`. The individual b-tag bins and OS/SS or
tau pass/fail alternatives are disjoint, but overlapping regions must not enter a simultaneous
fit as independent Poisson terms.
