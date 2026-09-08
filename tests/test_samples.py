from collections import Counter

from bbtautau_shapesmith.samples import EXPECTED_GROUP_SIZES, group_of, samples
from tests.helpers import REPO, nicks

DATABASE = REPO / "tests" / "fixtures" / "datasets.json"


def test_every_produced_nick_has_a_group():
    table = samples(DATABASE)
    assert sorted(s.nick for s in table) == sorted(nicks())
    assert Counter(s.group for s in table) == EXPECTED_GROUP_SIZES


def test_data_is_channel_specific():
    assert group_of("SingleMuon_Run2018A-UL2018") == ("data", ("mt",))
    assert group_of("EGamma_Run2018B-UL2018") == ("data", ("et",))
    assert group_of("Tau_Run2018C-UL2018") == ("data", ("tt",))
    assert group_of("TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv15-150X") == ("TT", None)
    assert all(s.kind == "data" for s in samples(DATABASE) if s.group == "data")
    assert all(s.kind == "mc" for s in samples(DATABASE) if s.group != "data")


def test_normalisation_values_are_present():
    table = samples(DATABASE)
    tt = next(s for s in table if s.nick.startswith("TTTo2L2Nu"))
    assert tt.xsec > 80 and tt.nevents > 1e8 and 0.9 < tt.generator_weight < 1.0
    hh = next(s for s in table if s.group == "HH2B2Tau")
    assert hh.xsec == 0.002246 and hh.nevents == 400000
