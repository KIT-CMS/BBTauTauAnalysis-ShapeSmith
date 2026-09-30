import pytest
from shapesmith.model import ABCD, AnalysisError, DataMinus, TemplateShift
from shapesmith.skim import needed_columns
from shapesmith.validate import validate

from bbtautau_shapesmith.analysis import build
from bbtautau_shapesmith.constants import FF_COLUMNS, NN_COLUMNS, TAU_CHANNELS
from bbtautau_shapesmith.systematics import embedding_variation_sums
from tests.helpers import EMBEDDING_LIST, branches, config, embedding_branches

COMBINATIONS = [(jet_fakes, embedding) for jet_fakes in ("mc", "ff") for embedding in (False, True)]
T_PARTS = {"ZTT", "TTT", "STT", "VVT", "TTVT", "EWKT"}
L_PARTS = {"ZL", "TTL", "STL", "VVL", "TTVL", "EWKL"}
J_PARTS = {"ZJ", "TTJ", "STJ", "VVJ", "TTVJ", "EWKJ"}
SINGLE_HIGGS = {"ggH125", "qqH125", "ttH125", "VH125"}


def analysis_for(jet_fakes="mc", embedding=False, **switches):
    sample_lists = ["sm2018_binned_v3", EMBEDDING_LIST] if embedding else ["sm2018_binned_v3"]
    analysis = build(config(jet_fakes=jet_fakes, embedding=embedding, sample_lists=sample_lists, **switches))
    validate(analysis)
    return analysis


@pytest.mark.parametrize("jet_fakes,embedding", COMBINATIONS)
@pytest.mark.parametrize("nn_friend", [False, True])
def test_every_switch_combination_builds_and_validates(jet_fakes, embedding, nn_friend):
    analysis = analysis_for(jet_fakes, embedding, nn_friend=nn_friend)
    for channel in analysis.channels.values():
        assert bool(channel.categories) == nn_friend


@pytest.mark.parametrize("jet_fakes,embedding", COMBINATIONS)
def test_process_table(jet_fakes, embedding):
    channel = analysis_for(jet_fakes, embedding).channel("mt")
    roles = {p.name: p.role for p in channel.processes}
    backgrounds = {name for name, role in roles.items() if role == "background"}
    genuine = {"EMB"} if embedding else T_PARTS
    jet_fakes_mc = J_PARTS | {"W"} if jet_fakes == "mc" else set()
    assert backgrounds == genuine | L_PARTS | jet_fakes_mc | SINGLE_HIGGS
    assert roles["data"] == "data" and roles["HH2B2Tau"] == "signal"
    assert ("TTT" in roles and roles["TTT"] == "auxiliary") == embedding  # the template of the ttbar contamination
    output = "jetFakes" if jet_fakes == "ff" else "QCD"
    assert set(channel.backgrounds()) == backgrounds | {output}
    top = {p.name for p in channel.processes if "top_pt" in p.selection.weights}
    assert top == {"TTT", "TTL", "TTJ"} & set(roles)  # only ttbar ntuples carry topPtReweightWeight


@pytest.mark.parametrize("jet_fakes,embedding", COMBINATIONS)
def test_estimators(jet_fakes, embedding):
    channel = analysis_for(jet_fakes, embedding).channel("tt")
    first, *rest = channel.estimators
    genuine = ("EMB",) if embedding else ("ZTT", "TTT", "STT", "VVT", "TTVT", "EWKT")
    lepton_fakes = ("ZL", "TTL", "STL", "VVL", "TTVL", "EWKL")
    if jet_fakes == "ff":
        assert first == DataMinus("jetFakes", "anti_iso", genuine + lepton_fakes)
    else:
        assert first == ABCD("QCD", "abcd_anti_iso", "abcd_same_sign", "abcd_same_sign_anti_iso",
                             genuine + lepton_fakes + ("ZJ", "TTJ", "STJ", "VVJ", "TTVJ", "EWKJ", "W"))
    sums = list(embedding_variation_sums("tt")) if embedding else []
    assert rest == sums + ([TemplateShift("CMS_htt_emb_ttbar_2018", "EMB", "TTT", 0.1)] if embedding else [])
    assert all(e not in analysis_for(jet_fakes, embedding, shape_systematics=False).channel("tt").estimators for e in sums)


@pytest.mark.parametrize("jet_fakes,embedding", COMBINATIONS)
@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_all_needed_columns_exist(jet_fakes, embedding, channel):
    definition = analysis_for(jet_fakes, embedding, nn_friend=True).channel(channel)
    friends = NN_COLUMNS | (set(FF_COLUMNS.values()) if jet_fakes == "ff" else set())
    for sample in definition.samples:
        available = embedding_branches(channel) if sample.kind == "embedding" else branches(channel) | {"npartons"}
        missing = needed_columns(definition, sample) - available - friends
        assert not missing, (sample.nick, missing)


def test_ml_export_labels():
    ml = analysis_for("ff", embedding=True).ml
    assert "TTT" not in ml.processes and "data" not in ml.processes
    assert ml.label_of["EMB"] == "is_DY" and ml.label_of["jetFakes"] == "is_jetFakes" and ml.region_of == {"jetFakes": "anti_iso"}
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


def test_embedding_needs_the_embedding_sample_list():
    with pytest.raises(AnalysisError, match="EMB"):
        validate(build(config(embedding=True)))


def test_style_covers_all_groups():
    analysis = analysis_for("mc", embedding=True)
    channel = analysis.channel("mt")
    groups = {channel.process(name).plot_group for name in channel.backgrounds() if name != "QCD"} | {"QCD", "jetFakes", "EMB", analysis.signal}
    assert groups <= set(analysis.style.colors)
    assert set(analysis.style.group_order) >= groups - {analysis.signal}
    for variable in channel.variables:
        assert variable in analysis.style.axis_labels["mt"]
