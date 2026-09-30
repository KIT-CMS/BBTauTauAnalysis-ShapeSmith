# Patches

## TauIDSFMeasurement-tauID_ES.patch

The tau-ID/ES measurement (`configs/tau_id_es_2018.yaml`) runs `MorphingTauID2017` of
[KIT-CMS/TauIDSFMeasurement](https://github.com/KIT-CMS/TauIDSFMeasurement), branch `tauID_ES` (84095dd), with jvoss's
uncommitted local changes. Without them the options `--tes_precision`, `--es_min` and `--es_max`, the categories DM1011
and DM*_PT20_40 / DM*_PT40_200, and the 2018 luminosity uncertainty 1.0084 do not exist. The patch is
`git diff` of `bin/MorphingTauID2017.cpp` and `src/HttSystematics_TauIDRun2.cc` in
`/work/jvoss/smhtt_ul_SFs_v15/CMSSW_14_1_0_pre4/src/CombineHarvester/TauIDSFMeasurement` (2026-09-30, unchanged since the
build of 2026-07-05), sha256 `b5fad3bf76b4465d4d49c912ffa9801a50b98730e0fa988d32290fa39cda0ca5`.

Until an own area exists, `combine.cmssw_dir` points to jvoss's built area, used read-only (all outputs are written below
the measurement's output directory). `MorphingTauID2017` writes its `cb.PrintAll()` log to the hard-coded path
`/work/jvoss/ntuples/smhtt_ul_SFs_v15/log/cb_PrintAll.log`; that directory does not exist, so the binary prints
"Could not open log file" and carries on.

An own area, as jvoss's (HTTPS clones; SSH is not needed):

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
scram b -j 8
```

jvoss's area also carries CombineHarvester/SMRun2Legacy (not needed here) and local changes to
HiggsAnalysis/CombinedLimit `scripts/plotGof.py` and `scripts/plotImpacts.py` (GoF and impacts, not part of the
measurement stage).
