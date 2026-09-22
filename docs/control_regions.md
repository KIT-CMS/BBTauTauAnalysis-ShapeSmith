# Control regions

Both Analyses provide raw data/MC comparisons next to the nominal selection. They need no
production of their own: the `sm2018_binned_v4` ntuples (CROWN `sm_config`, all six scopes) keep
both charges, VVVLoose taus, zero b-tags and, in et/mt, the light lepton up to an isolation of 0.5.

## Tau channels: `configs/sm2018_binned_v4.yaml` with `control_regions: true`

```bash
shapesmith hist -c configs/sm2018_binned_v4.yaml --control --regions all --skip-systematics
shapesmith plot -c configs/sm2018_binned_v4.yaml --control --channels et,mt --region w_highmt_pass --variables mt_1,met,pt_2,n_bjets
shapesmith plot -c configs/sm2018_binned_v4.yaml --control --channels et,mt --region w_highmt_fail --variables mt_1,met,pt_2
shapesmith plot -c configs/sm2018_binned_v4.yaml --control --region btag2_os_pass --variables m_vis,met,n_bjets
shapesmith plot -c configs/sm2018_binned_v4.yaml --control --region btag1_ss_fail --variables pt_2,m_vis
shapesmith plot -c configs/sm2018_binned_v4.yaml --control --channels et,mt --region lepton_antiiso --variables iso_1,mt_1
```

| Region | Replacement relative to the nominal tau baseline |
|---|---|
| `btag{0,1,2,3p}_{os,ss}_{pass,fail}` | ≥2 jets, stated b-tag bin and charge; Medium tau pass, or VVVLoose-but-not-Medium fail |
| `w_highmt_pass`, `w_highmt_fail` (et/mt) | OS, zero b-tags, ≥2 jets, mT(lepton, MET) > 80 GeV; tau pass/fail |
| `lepton_antiiso`, `lepton_antiiso_ss` (et/mt) | 0.15 ≤ lepton isolation < 0.5; OS or SS; other nominal cuts unchanged |

In tt the fail selection requires exactly one Medium leg to fail while both pass VVVLoose. The
anti-electron/muon tau requirements, trigger cuts and extra-lepton vetoes remain. The switch widens
the skim (both charges, VVVLoose taus, ≥2 jets or the nominal b-tag requirement, no mT cut, lepton
isolation < 0.5) and leaves the nominal selection unchanged, so one skim serves nominal shapes and
controls. The zero-b regions do not require a reconstructed b pair; CROWN's b-pair quantities are
sentinel values there, so plot lepton, jet and MET variables.

The regions require `jet_fakes: mc` and carry no data-driven QCD or jetFakes estimate: a residual
in a non-nominal region is not automatically an MC defect. Fail regions apply the passing tau-ID SF
only to passing genuine-tau legs; failed genuine taus stay uncorrected. The muon anti-isolation
sideband does not use the isolation SF. The nominal plot with the ABCD QCD estimate needs
`shapesmith estimate ... --control` first.

## Light-dilepton channels: `configs/sm2018_binned_v4_dilepton.yaml`

```bash
shapesmith hist -c configs/sm2018_binned_v4_dilepton.yaml --control --regions all --skip-systematics
shapesmith plot -c configs/sm2018_binned_v4_dilepton.yaml --control --channels em --region btag2 --variables yield,pt_1,pt_2,met,n_bjets
shapesmith plot -c configs/sm2018_binned_v4_dilepton.yaml --control --channels mm,ee --region nominal --variables yield,m_vis,pt_vis,n_jets
shapesmith plot -c configs/sm2018_binned_v4_dilepton.yaml --control --channels mm,ee --region btag1 --variables yield,m_vis,pt_vis
```

| Channel | Baseline | Data | Purpose |
|---|---|---|---|
| em | isolated OS electron (leg 1) and muon (leg 2), pT > 15/26 GeV, single-muon trigger, ≥2 jets | SingleMuon | ttbar kinematics and b-tag multiplicity |
| mm | isolated OS muons, pT > 26/15 GeV, single-muon trigger, 70 < m(ll) < 110 GeV | SingleMuon | Z normalisation, recoil, Z+heavy flavour |
| ee | isolated OS electrons, pT > 34/15 GeV, single-electron trigger, 70 < m(ll) < 110 GeV | EGamma | electron cross-check of Z+jets |

The 26/34 GeV cuts match the offline thresholds inside CROWN's trigger flags; both leptons have
isolation < 0.15 and extra-lepton vetoes. CROWN writes every data stream into every scope, so
`samples.py` routes SingleMuon to em/mm and EGamma to ee. `nominal` is inclusive in b-tags (and,
for the Z channels, in jets); `btag0`, `btag1`, `btag2`, `btag3p` require ≥2 jets and partition the
events by b-tag count; `same_sign` flips the charge requirement for a qualitative look at nonprompt
backgrounds. MC is unsplit (no gen-match selections, no fake estimate); both lepton legs receive
ID/isolation or reconstruction weights, the trigger SF belongs to the triggering leg, TT keeps its
top-pT weight. Only the correlated/uncorrelated b-tag variations are booked (omit
`--skip-systematics` to fill them). The one-bin `yield` variable counts every selected event;
kinematic histograms have finite ranges and are no substitute for it.

## Reading the comparisons

Start with em `btag1`/`btag2` (yield, lepton pT, MET, b-tag multiplicity), then the inclusive Z
yields and pT(ll) in ee/mm before their b-tag bins, then the tau W high-mT pass/fail and SS regions.
Agreement in em with a discrepancy only in the tau channels points to tau or fake modelling; a
common b-tag trend points to the tagging or the flavour composition. The regions overlap with each
other and with the nominal selection, so they must not enter a simultaneous fit as independent
Poisson terms.
