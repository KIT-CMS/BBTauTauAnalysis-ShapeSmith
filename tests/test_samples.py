import pytest
from shapesmith.model import AnalysisError
from shapesmith.skim import needed_columns

from bbtautau_shapesmith.analysis import build as build_tau
from bbtautau_shapesmith.constants import DILEPTON_CHANNELS, TAU_CHANNELS
from bbtautau_shapesmith.dilepton import build as build_dilepton
from bbtautau_shapesmith.samples import route, samples
from tests.helpers import DATABASE, EMBEDDING_LIST, branches, config, nicks

INCLUSIVE_DY = "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1"
CHANNELS = list(TAU_CHANNELS + DILEPTON_CHANNELS)


def test_every_produced_nick_has_a_group_and_its_kind_from_the_database():
    by_channel = samples(DATABASE, ["sm2018_binned_v2"], CHANNELS)
    assert {s.nick for chosen in by_channel.values() for s in chosen} == set(nicks())
    table = {s.nick: s for chosen in by_channel.values() for s in chosen}
    assert all((s.kind == "data") == (s.group == "data") and s.kind in ("data", "mc") for s in table.values())


def test_data_streams_are_routed_to_the_channels_they_trigger():
    assert route("SingleMuon_Run2018A-UL2018") == ("data", ("mt", "em", "mm"))
    assert route("EGamma_Run2018B-UL2018") == ("data", ("et", "ee"))
    assert route("Tau_Run2018C-UL2018") == ("data", ("tt",))
    assert route("TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv15-150X") == ("TT", None)


def test_embedding_nicks_are_routed_by_their_final_state():
    routes = {nick: route(nick) for nick in nicks(EMBEDDING_LIST)}
    assert sorted(channels for _, channels in routes.values()) == [("et",)] * 4 + [("mt",)] * 4 + [("tt",)] * 4
    assert {group for group, _ in routes.values()} == {"EMB"}
    assert routes["EmbeddingRun2018C_cwinter-embedding_2018UL_mutau_2018C_1772450480-00000000000000000000000000000000"] == ("EMB", ("mt",))
    assert route("EmbeddingRun2018A_cwinter-embedding_2018UL_muemb_A_1772534952-00000000000000000000000000000000") == ("MUEMB", ("mm",))
    with pytest.raises(KeyError, match="final states"):
        route("EmbeddingRun2018A_cwinter-embedding_2018UL_elmu_A_1772534952")


def test_embedding_samples_go_to_their_channel_only():
    by_channel = samples(DATABASE, ["sm2018_binned_v2", EMBEDDING_LIST], CHANNELS)
    for channel in CHANNELS:
        embedded = [s for s in by_channel[channel] if s.kind == "embedding"]
        assert len(embedded) == (4 if channel in TAU_CHANNELS else 0)
        assert all(s.group == "EMB" and s.norm_weight == 1.0 for s in embedded)


def test_a_nick_in_two_sample_lists_raises():
    with pytest.raises(AnalysisError, match="more than one"):
        samples(DATABASE, ["sm2018_binned_v2", "sm2018_binned_v3"], ["mt"])


def test_missing_sample_list_is_reported():
    with pytest.raises(FileNotFoundError, match="missing_sample_list"):
        build_tau(config(sample_lists=["missing_sample_list"]))


def test_v3_adds_the_zero_parton_part_of_the_inclusive_dy_sample():
    assert sorted(nicks("sm2018_binned_v3")) == sorted([*nicks("sm2018_binned_v2"), INCLUSIVE_DY])
    table = {s.nick: s for s in samples(DATABASE, ["sm2018_binned_v3"], ["mt"])["mt"]}
    inclusive = table[INCLUSIVE_DY]
    assert (inclusive.group, inclusive.cut) == ("DY", "npartons == 0")
    assert inclusive.norm_weight == pytest.approx(6077.22 / (196626007 * 0.6729628751500812))  # the whole dataset, not its 0-parton part
    binned = [s for s in table.values() if s.nick.startswith("DYJetsToLL_LHEFilterPtZ-")]
    assert len(binned) == 6 and all(s.cut is None for s in binned)
    assert [s.nick for s in table.values() if s.cut is not None] == [INCLUSIVE_DY]


@pytest.mark.parametrize("channel", CHANNELS)
def test_only_the_inclusive_dy_sample_needs_npartons(channel):
    # The branch fixtures come from a ttbar ntuple, which has no npartons (CROWN writes it for DY, W+jets and EWK samples only),
    # so it is the one column of sm2018_binned_v3 they cannot confirm. Only the inclusive DY sample reads it: the columns of
    # every other sample, and so the contracts of their existing skims, are those of sm2018_binned_v2.
    build = build_tau if channel in TAU_CHANNELS else build_dilepton
    analysis = "bbtautau_shapesmith.analysis:build" if channel in TAU_CHANNELS else "bbtautau_shapesmith.dilepton:build"
    v2, v3 = (build(config([channel], analysis, sample_lists=[name])).channel(channel) for name in ("sm2018_binned_v2", "sm2018_binned_v3"))
    columns_v2 = {s.nick: needed_columns(v2, s) for s in v2.samples}
    for sample in v3.samples:
        columns = needed_columns(v3, sample)
        if sample.nick == INCLUSIVE_DY:
            assert columns - branches(channel) == {"npartons"}
        else:
            assert columns == columns_v2[sample.nick] and columns <= branches(channel)
