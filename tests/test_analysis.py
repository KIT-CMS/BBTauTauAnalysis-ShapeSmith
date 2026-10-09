import pytest
from shapesmith.model import ABCD, AnalysisError, DataMinus, TemplateShift, Variable
from shapesmith.skim import needed_columns
from shapesmith.validate import validate

from bbtautau_shapesmith import dilepton, ff_measurement
from bbtautau_shapesmith.analysis import build, process_union
from bbtautau_shapesmith.constants import FF_COLUMNS, FF_RAW_COLUMNS, LT_CHANNELS, NN_COLUMNS, TAU_CHANNELS
from bbtautau_shapesmith.systematics import embedding_variation_sums
from bbtautau_shapesmith.weights import dy_njets_btag_weight, fake_factor_weight
from tests.helpers import EMBEDDING_LIST, branches, config, embedded, embedding_branches

COMBINATIONS = [(jet_fakes, embedding) for jet_fakes in ("mc", "ff") for embedding in (False, True, ["et", "mt"])]
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


@pytest.mark.parametrize("nn_friend", [False, True])
def test_control_fit_categories(nn_friend):
    """fit_categories: control gives every tau channel one category with the nominal region in one bin."""
    analysis = analysis_for("ff", ["et", "mt"], fit_categories="control", nn_friend=nn_friend)
    for channel in analysis.channels.values():
        (category,) = channel.categories
        assert (category.name, category.cut, category.variable) == ("nominal", "True", Variable("yield", "1.0", (0.5, 1.5)))
        assert any(needed_columns(channel, sample) & NN_COLUMNS for sample in channel.samples) == nn_friend  # NN_score as a control variable


@pytest.mark.parametrize("jet_fakes,embedding", COMBINATIONS)
def test_process_table(jet_fakes, embedding):
    channel = analysis_for(jet_fakes, embedding).channel("mt")
    roles = {p.name: p.role for p in channel.processes}
    backgrounds = {name for name, role in roles.items() if role == "background"}
    genuine = {"EMB"} if embedded(embedding, "mt") else T_PARTS
    jet_fakes_mc = J_PARTS | {"W"} if jet_fakes == "mc" else set()
    assert backgrounds == genuine | L_PARTS | jet_fakes_mc | SINGLE_HIGGS
    assert roles["data"] == "data" and roles["HH2B2Tau"] == "signal"
    assert ("TTT" in roles and roles["TTT"] == "auxiliary") == embedded(embedding, "mt")  # the template of the ttbar contamination
    output = "jetFakes" if jet_fakes == "ff" else "QCD"
    assert set(channel.backgrounds()) == backgrounds | {output}
    top = {p.name for p in channel.processes if "top_pt" in p.selection.weights}
    assert top == {"TTT", "TTL", "TTJ"} & set(roles)  # only ttbar ntuples carry topPtReweightWeight


@pytest.mark.parametrize("jet_fakes,embedding", COMBINATIONS)
def test_estimators(jet_fakes, embedding):
    channel = analysis_for(jet_fakes, embedding).channel("tt")
    first, *rest = channel.estimators
    genuine = ("EMB",) if embedded(embedding, "tt") else ("ZTT", "TTT", "STT", "VVT", "TTVT", "EWKT")
    lepton_fakes = ("ZL", "TTL", "STL", "VVL", "TTVL", "EWKL")
    if jet_fakes == "ff":
        assert first == DataMinus("jetFakes", "anti_iso", genuine + lepton_fakes)
    else:
        assert first == ABCD("QCD", "abcd_anti_iso", "abcd_same_sign", "abcd_same_sign_anti_iso",
                             genuine + lepton_fakes + ("ZJ", "TTJ", "STJ", "VVJ", "TTVJ", "EWKJ", "W"))
    sums = list(embedding_variation_sums("tt")) if embedded(embedding, "tt") else []
    assert rest == sums + ([TemplateShift("CMS_htt_emb_ttbar_2018", "EMB", "TTT", 0.1)] if embedded(embedding, "tt") else [])
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


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_raw_fake_factors_read_only_the_raw_friend_columns(channel):
    raw = analysis_for("ff", embedding=True, ff_type="raw", shape_systematics=False)
    corrected = analysis_for("ff", embedding=True, shape_systematics=False)
    definition = raw.channel(channel)
    assert definition.region("anti_iso").add_weights == {"fake_factor": fake_factor_weight(channel, "raw")}
    assert definition.processes == corrected.channel(channel).processes and definition.estimators == corrected.channel(channel).estimators
    for sample in definition.samples:
        needed = needed_columns(definition, sample)
        assert not needed & set(FF_COLUMNS.values()), sample.nick
        available = embedding_branches(channel) if sample.kind == "embedding" else branches(channel) | {"npartons"}
        assert not needed - available - set(FF_RAW_COLUMNS.values()), sample.nick
    data = next(s for s in definition.samples if s.kind == "data")
    legs = ("lt",) if channel in LT_CHANNELS else ("tt_1", "tt_2")
    assert needed_columns(definition, data) & set(FF_RAW_COLUMNS.values()) == {FF_RAW_COLUMNS[leg] for leg in legs}


def test_embedding_per_channel():
    """embedding: [et, mt]: EMB and the ttbar contamination in et and mt, the MC T parts in tt."""
    analysis = analysis_for("mc", ["et", "mt"])
    for name in LT_CHANNELS:
        channel = analysis.channel(name)
        assert "EMB" in channel.backgrounds() and channel.process("TTT").role == "auxiliary" and not T_PARTS & set(channel.backgrounds())
        assert TemplateShift("CMS_htt_emb_ttbar_2018", "EMB", "TTT", 0.1) in channel.estimators
    tt = analysis.channel("tt")
    assert T_PARTS <= set(tt.backgrounds()) and "EMB" not in {p.name for p in tt.processes}
    assert not any(s.kind == "embedding" for s in tt.samples)
    assert not any(isinstance(e, TemplateShift) for e in tt.estimators) and not any("emb" in v.name for v in tt.variations)
    lnn = {(entry.name, entry.channels): set(entry.processes) for entry in analysis.lnn}
    assert {"ZTT", "TTT", "STT", "VVT"} <= lnn[("lumi_13TeV_$ERA", None)] and "EMB" not in lnn[("lumi_13TeV_$ERA", None)]
    assert lnn[("htt_tjXsec", None)] == {"TTT", "TTL", "TTJ"} and lnn[("htt_zjXsec", None)] == {"ZTT", "ZL", "ZJ"}
    ml = analysis.ml
    assert {"EMB", "ZTT", "TTT"} <= set(ml.processes)
    assert "EMB" in ml.exported(analysis.channel("et")) and not {"ZTT", "TTT"} & set(ml.exported(analysis.channel("et")))
    assert {"ZTT", "TTT"} <= set(ml.exported(tt)) and "EMB" not in ml.exported(tt)


def test_dy_correction_weights_the_dy_processes_only():
    def corrected(analysis):
        return {(name, p.name) for name, channel in analysis.channels.items() for p in channel.processes if "dy_correction" in p.selection.weights}

    tau = analysis_for("mc", dy_correction="njets_btag")
    assert corrected(tau) == {(name, p) for name in TAU_CHANNELS for p in ("ZTT", "ZL", "ZJ")}
    assert tau.channel("mt").process("ZL").selection.weights["dy_correction"] == dy_njets_btag_weight()
    assert not corrected(analysis_for("mc"))
    fake_factors = ff_measurement.build(config(analysis="bbtautau_shapesmith.ff_measurement:build", sample_lists=["sm2018_binned_v3"], dy_correction="njets_btag"))
    assert corrected(fake_factors) == {(name, p) for name in TAU_CHANNELS for p in ("ZTT", "ZL", "ZJ")}
    controls = dilepton.build(config(["em", "mm", "ee"], "bbtautau_shapesmith.dilepton:build", dy_correction="njets_btag"))
    assert corrected(controls) == {(name, "DY") for name in ("em", "mm", "ee")}


def test_ml_export_labels():
    ml = analysis_for("ff", embedding=True).ml
    assert "TTT" not in ml.processes and "data" not in ml.processes
    assert ml.label_of["EMB"] == "is_DY" and ml.label_of["jetFakes"] == "is_jetFakes" and ml.region_of == {"jetFakes": "anti_iso"}
    labels = analysis_for("mc").ml.label_of
    assert labels["EWKJ"] == labels["W"] == "is_jetFakes" and labels["EWKT"] == "is_Other" and labels["ZTT"] == "is_DY"


def test_processes_of_every_channel():
    """lnN and ML labels come from the union of the channel tables: TTT is auxiliary with embedding, a background without."""
    embedded, simulated = analysis_for("mc", embedding=True).channel("mt"), analysis_for("mc").channel("tt")
    roles = {p.name: p.role for p in process_union([embedded, simulated])}
    assert roles["EMB"] == roles["ZTT"] == roles["TTT"] == "background"
    assert [p.name for p in process_union([embedded, simulated])] == list(dict.fromkeys(p.name for p in embedded.processes + simulated.processes))


@pytest.mark.parametrize("switches,match", [
    ({"sample_list": "sm2018_binned_v2"}, "sample_list"),
    ({"production": "sm2018_binned_v1"}, "production"),
    ({"embedding": "yes"}, "embedding"),
    ({"embedding": ["xx"]}, "embedding"),
    ({"embedding": ["et", "et"]}, "listed twice"),
    ({"jet_fakes": "data"}, "jet_fakes"),
    ({"sample_lists": []}, "sample_lists"),
    ({"control_regions": True, "jet_fakes": "ff"}, "control_regions requires jet_fakes: mc"),
    ({"ff_type": "uncorrected", "jet_fakes": "ff"}, "ff_type"),
    ({"ff_type": "raw", "shape_systematics": False}, "ff_type raw requires jet_fakes: ff"),
    ({"ff_type": "raw", "jet_fakes": "ff"}, "ff_type raw supports nominal control plots only; set shape_systematics: false"),
])
def test_invalid_switches_are_reported(switches, match):
    with pytest.raises(AnalysisError, match=match):
        build(config(**switches))


def test_an_embedded_channel_must_be_a_channel_of_the_run():
    with pytest.raises(AnalysisError, match="embedding: tt not among the channels et, mt"):
        build(config(["et", "mt"], embedding=["tt"], sample_lists=["sm2018_binned_v3", EMBEDDING_LIST]))


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
