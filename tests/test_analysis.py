import pytest
from shapesmith.model import ABCD, AnalysisError, DataMinus
from shapesmith.skim import needed_columns
from shapesmith.validate import validate

from bbtautau_shapesmith.analysis import build
from bbtautau_shapesmith.constants import FF_COLUMNS, NN_COLUMNS, TAU_CHANNELS
from tests.helpers import branches, config

T_PARTS = {"ZTT", "TTT", "STT", "VVT", "TTVT", "EWKT"}
L_PARTS = {"ZL", "TTL", "STL", "VVL", "TTVL", "EWKL"}
J_PARTS = {"ZJ", "TTJ", "STJ", "VVJ", "TTVJ", "EWKJ"}
SINGLE_HIGGS = {"ggH125", "qqH125", "ttH125", "VH125"}


def analysis_for(jet_fakes="mc", **switches):
    analysis = build(config(jet_fakes=jet_fakes, sample_lists=["sm2018_binned_v3"], **switches))
    validate(analysis)
    return analysis


@pytest.mark.parametrize("jet_fakes", ["mc", "ff"])
@pytest.mark.parametrize("nn_friend", [False, True])
def test_every_switch_combination_builds_and_validates(jet_fakes, nn_friend):
    analysis = analysis_for(jet_fakes, nn_friend=nn_friend)
    for channel in analysis.channels.values():
        assert bool(channel.categories) == nn_friend


@pytest.mark.parametrize("jet_fakes", ["mc", "ff"])
def test_process_table(jet_fakes):
    channel = analysis_for(jet_fakes).channel("mt")
    roles = {p.name: p.role for p in channel.processes}
    backgrounds = {name for name, role in roles.items() if role == "background"}
    jet_fakes_mc = J_PARTS | {"W"} if jet_fakes == "mc" else set()
    assert backgrounds == T_PARTS | L_PARTS | jet_fakes_mc | SINGLE_HIGGS
    assert roles["data"] == "data" and roles["HH2B2Tau"] == "signal"
    output = "jetFakes" if jet_fakes == "ff" else "QCD"
    assert set(channel.backgrounds()) == backgrounds | {output}
    top = {p.name for p in channel.processes if "top_pt" in p.selection.weights}
    assert top == {"TTT", "TTL", "TTJ"} & set(roles)  # only ttbar ntuples carry topPtReweightWeight


@pytest.mark.parametrize("jet_fakes", ["mc", "ff"])
def test_estimators(jet_fakes):
    (estimator,) = analysis_for(jet_fakes).channel("tt").estimators
    genuine_and_lepton_fakes = ("ZTT", "TTT", "STT", "VVT", "TTVT", "EWKT", "ZL", "TTL", "STL", "VVL", "TTVL", "EWKL")
    if jet_fakes == "ff":
        assert estimator == DataMinus("jetFakes", "anti_iso", genuine_and_lepton_fakes)
    else:
        assert estimator == ABCD("QCD", "abcd_anti_iso", "abcd_same_sign", "abcd_same_sign_anti_iso",
                                 genuine_and_lepton_fakes + ("ZJ", "TTJ", "STJ", "VVJ", "TTVJ", "EWKJ", "W"))


@pytest.mark.parametrize("jet_fakes", ["mc", "ff"])
@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_all_needed_columns_exist(jet_fakes, channel):
    definition = analysis_for(jet_fakes, nn_friend=True).channel(channel)
    friends = NN_COLUMNS | (set(FF_COLUMNS.values()) if jet_fakes == "ff" else set())
    for sample in definition.samples:
        missing = needed_columns(definition, sample) - branches(channel) - {"npartons"} - friends
        assert not missing, (sample.nick, missing)


def test_ml_export_labels():
    ml = analysis_for("ff").ml
    assert "data" not in ml.processes and ml.label_of["jetFakes"] == "is_jetFakes" and ml.region_of == {"jetFakes": "anti_iso"}
    labels = analysis_for("mc").ml.label_of
    assert labels["EWKJ"] == labels["W"] == "is_jetFakes" and labels["EWKT"] == "is_Other" and labels["ZTT"] == "is_DY"


@pytest.mark.parametrize("switches,match", [
    ({"sample_list": "sm2018_binned_v2"}, "sample_list"),
    ({"production": "sm2018_binned_v1"}, "production"),
    ({"embedding": "yes"}, "embedding"),
    ({"jet_fakes": "data"}, "jet_fakes"),
    ({"sample_lists": []}, "sample_lists"),
    ({"control_regions": True, "jet_fakes": "ff"}, "control_regions requires jet_fakes: mc"),
])
def test_invalid_switches_are_reported(switches, match):
    with pytest.raises(AnalysisError, match=match):
        build(config(**switches))


def test_embedding_needs_embedded_samples():
    with pytest.raises(AnalysisError, match="EMB"):
        validate(build(config(embedding=True)))


def test_style_covers_all_groups():
    analysis = analysis_for("mc")
    channel = analysis.channel("mt")
    groups = {channel.process(name).plot_group for name in channel.backgrounds() if name != "QCD"} | {"QCD", "jetFakes", "EMB", analysis.signal}
    assert groups <= set(analysis.style.colors)
    assert set(analysis.style.group_order) >= groups - {analysis.signal}
    for variable in channel.variables:
        assert variable in analysis.style.axis_labels["mt"]
