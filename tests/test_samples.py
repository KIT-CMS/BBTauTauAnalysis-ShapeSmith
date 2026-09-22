from bbtautau_shapesmith.samples import group_of, samples
from tests.helpers import DATABASE, nicks


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
