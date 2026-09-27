import pytest
from shapesmith.config import RunConfig
from shapesmith.skim import columns_for_sample, required_columns

from bbtautau_shapesmith.analysis import build as build_tau
from bbtautau_shapesmith.constants import DILEPTON_CHANNELS, TAU_CHANNELS
from bbtautau_shapesmith.dilepton_analysis import build as build_dilepton
from bbtautau_shapesmith.samples import group_of, samples
from tests.helpers import DATABASE, branches, nicks


def test_every_produced_nick_has_a_group():
    table = samples(DATABASE)
    assert sorted(s.nick for s in table) == sorted(nicks())


def test_data_streams_are_routed_to_the_channels_they_trigger():
    assert group_of("SingleMuon_Run2018A-UL2018") == ("data", ("mt", "em", "mm"))
    assert group_of("EGamma_Run2018B-UL2018") == ("data", ("et", "ee"))
    assert group_of("Tau_Run2018C-UL2018") == ("data", ("tt",))
    assert group_of("TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv15-150X") == ("TT", None)
    assert all(s.kind == "data" for s in samples(DATABASE) if s.group == "data")
    assert all(s.kind == "mc" for s in samples(DATABASE) if s.group != "data")


INCLUSIVE_DY = "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1"


def test_v3_adds_the_zero_parton_part_of_the_inclusive_dy_sample():
    assert sorted(nicks("sm2018_binned_v3")) == sorted([*nicks("sm2018_binned_v2"), INCLUSIVE_DY])
    table = {s.nick: s for s in samples(DATABASE, "sm2018_binned_v3")}
    inclusive = table[INCLUSIVE_DY]
    assert (inclusive.group, inclusive.cut) == ("DY", "npartons == 0")
    assert inclusive.norm_weight == pytest.approx(6077.22 / (196626007 * 0.6729628751500812))  # the whole dataset, not its 0-parton part
    binned = [s for s in table.values() if s.nick.startswith("DYJetsToLL_LHEFilterPtZ-")]
    assert len(binned) == 6 and all(s.cut is None for s in binned)
    assert [s.nick for s in table.values() if s.cut is not None] == [INCLUSIVE_DY]


@pytest.mark.parametrize("channel", TAU_CHANNELS + DILEPTON_CHANNELS)
def test_only_the_inclusive_dy_sample_needs_npartons(channel):
    # The branch fixtures come from a ttbar ntuple, which has no npartons (CROWN writes it for DY, W+jets and EWK samples only),
    # so it is the one column of sm2018_binned_v3 they cannot confirm. Only the inclusive DY sample reads it: the columns of
    # every other sample, and so the contracts of their existing skims, are those of sm2018_binned_v2.
    build = build_tau if channel in TAU_CHANNELS else build_dilepton
    v2, v3 = (build(RunConfig(analysis="unused", era="2018", channels=[channel], ntuples={"base": "/unused"}, skim_dir="/unused/skim",
                              output_dir="/unused/output", sample_database=DATABASE, switches={"sample_list": sample_list}))
              for sample_list in ("sm2018_binned_v2", "sm2018_binned_v3"))
    assert required_columns(v3, channel) == required_columns(v2, channel) <= branches(channel)
    inclusive = next(s for s in v3.samples if s.nick == INCLUSIVE_DY)
    assert columns_for_sample(v3, channel, inclusive) - branches(channel) == {"npartons"}
    assert [s.nick for s in v3.samples if "npartons" in columns_for_sample(v3, channel, s)] == [INCLUSIVE_DY]
