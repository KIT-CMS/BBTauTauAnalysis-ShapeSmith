# Patches

## TauIDSFMeasurement-tauID_ES.patch

The tau-ID/ES measurement (`configs/tau_id_es_2018.yaml`, `configs/smrun2_tau_id_v1_id_es.yaml`) runs
`MorphingTauID2017` of [KIT-CMS/TauIDSFMeasurement](https://github.com/KIT-CMS/TauIDSFMeasurement), branch `tauID_ES`
(84095dd), with jvoss's uncommitted local changes. Without them the options `--tes_precision`, `--es_min` and
`--es_max` and the categories DM1011 and DM*_PT20_40 / DM*_PT40_200 do not exist. The patch starts from `git diff` of
`bin/MorphingTauID2017.cpp` and `src/HttSystematics_TauIDRun2.cc` in
`/work/jvoss/smhtt_ul_SFs_v15/CMSSW_14_1_0_pre4/src/CombineHarvester/TauIDSFMeasurement` (2026-09-30, unchanged since the
build of 2026-07-05; sha256 of that diff `b5fad3bf76b4465d4d49c912ffa9801a50b98730e0fa988d32290fa39cda0ca5`), with these
changes to `HttSystematics_TauIDRun2.cc`:

- the regrouped jet energy scale sources of the unpatched branch
  (`CMS_scale_j_{Absolute,BBEC1,EC2,HF,RelativeSample}_$ERA`, `CMS_scale_j_{Absolute,BBEC1,EC2,HF,FlavorQCD,RelativeBal}`)
  instead of jvoss's 28 individual sources, since the CROWN variations package produces the regrouped set; JER stays
  commented out (2026-10-03);
- v7 (2026-10-17): no `CMS_scale_j_HEMIssue_$ERA` (2018 vetoes the HEM events, E29), `CMS_prefiring` in every era
  instead of all but 2018 (E11: 2018 has the muon part), and `lumi_13TeV_$ERA` the total of the LUM POG Cholesky
  scheme of the year (spec 6.6): 1.0123 (2016 halves), 1.0082 (2017), 1.0084 (2018). MorphingTauID2017 throws on a
  declared shape that the synced shapes lack (`CombineTools/src/TFileIO.cc:24`, "TH1 ... not found"), so the v7
  shapes (without HEM, with `CMS_prefiring` in 2018) need exactly this declaration.

The patch is `git diff bin/MorphingTauID2017.cpp` followed by `git diff -U100000 src/HttSystematics_TauIDRun2.cc`: the
second file with full context, so that `tests/test_tau_id_patch.py` reads the whole patched file and checks, per era,
that every shape it declares for a process of the measurement is one the analysis produces (and the lumi value). It
applies with `git apply` to a clean 84095dd. sha256 `1d15cde86f47981899689b9d33d2a5d2f87238af14b71dce1cbc2713944df61e`.

jvoss's built area declares the 28 individual sources, which the shapes do not have, so the measurement needs an own
area built with this patch; `combine.cmssw_dir` points to it (`/work/sdaigler/tau_id_es/CMSSW_14_1_0_pre4`, built
2026-10-03, rebuilt with the v7 changes on 2026-10-17: `scram b` in `CombineHarvester/TauIDSFMeasurement`; checked with
MorphingTauID2017 on the v5 2018 synced shapes of Tight/VVLoose, DM0, which now stops at the missing
`W_CMS_prefiringUp`, and on a copy without the HEM shapes and with `CMS_prefiring`, which gives the cards). `MorphingTauID2017` writes its `cb.PrintAll()` log to the hard-coded path
`/work/jvoss/ntuples/smhtt_ul_SFs_v15/log/cb_PrintAll.log`; that directory does not exist, so the binary prints
"Could not open log file" and carries on.

The own area, as jvoss's (HTTPS clones; SSH is not needed):

```bash
export SCRAM_ARCH=el9_amd64_gcc12
source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsrel CMSSW_14_1_0_pre4 && cd CMSSW_14_1_0_pre4/src && cmsenv
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
git -C HiggsAnalysis/CombinedLimit checkout v10.0.2
git clone https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
git -C CombineHarvester checkout v3.0.0
git clone -b tauID_ES https://github.com/KIT-CMS/TauIDSFMeasurement.git CombineHarvester/TauIDSFMeasurement
git -C CombineHarvester/TauIDSFMeasurement checkout 84095dd
git -C CombineHarvester/TauIDSFMeasurement apply <this repo>/patches/TauIDSFMeasurement-tauID_ES.patch
git clone -b ul https://github.com/KIT-CMS/SMRun2Legacy.git CombineHarvester/SMRun2Legacy
git -C CombineHarvester/SMRun2Legacy checkout f103cf5
scram b -j 8
```

CombineHarvester/SMRun2Legacy (branch `ul`, f103cf5 as in jvoss's area) is needed for the build:
TauIDSFMeasurement's `src/BinomialBinByBin.cc` includes `CombineHarvester/SMRun2Legacy/interface/BinomialBinByBin.h`.
Built this way on 2026-10-03 in `/work/sdaigler/tau_id_es/CMSSW_14_1_0_pre4`.

jvoss's area also carries local changes to
HiggsAnalysis/CombinedLimit `scripts/plotGof.py` and `scripts/plotImpacts.py` (GoF and impacts, not part of the
measurement stage).
