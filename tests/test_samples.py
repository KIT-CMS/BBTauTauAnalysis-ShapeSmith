from collections import Counter

from bbtautau_shapesmith.samples import group_of, samples
from tests.helpers import DATABASE, nicks


def test_every_produced_nick_has_a_group():
    table = samples(DATABASE)
    assert sorted(s.nick for s in table) == sorted(nicks())
    assert Counter(s.group for s in table) == Counter({"data": 12, "HH2B2Tau": 1, "DY": 7, "W": 3, "EWK": 7, "ST": 6, "TT": 3, "TTV": 11, "VV": 14, "ggH": 1, "qqH": 2, "ttH": 2, "VH": 10})


def test_data_streams_are_routed_to_the_channels_they_trigger():
    assert group_of("SingleMuon_Run2018A-UL2018") == ("data", ("mt", "em", "mm"))
    assert group_of("EGamma_Run2018B-UL2018") == ("data", ("et", "ee"))
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
