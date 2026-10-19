import pytest
from shapesmith.model import ABCD, AnalysisError, DataMinus, TemplateShift, Variable
from shapesmith.skim import needed_columns
from shapesmith.validate import validate

from bbtautau_shapesmith import dilepton, ff_measurement
from bbtautau_shapesmith.analysis import build, process_union
from bbtautau_shapesmith.constants import FF_COLUMNS, FF_RAW_COLUMNS, LT_CHANNELS, NN_COLUMNS, TAU_CHANNELS
from bbtautau_shapesmith.systematics import btag_variations, embedding_variation_sums, prefiring_variations
from bbtautau_shapesmith.variables import DIAGNOSTIC_VARIABLES, FASTMTT_VARIABLES
from bbtautau_shapesmith.weights import dy_njets_btag_weight, fake_factor_weight
from tests.helpers import EMBEDDING_LIST, V6_COLUMNS, branches, config, embedded, embedding_branches

COMBINATIONS = [(jet_fakes, embedding) for jet_fakes in ("mc", "ff") for embedding in (False, True, ["et", "mt"])]
T_PARTS = {"ZTT", "TTT", "STT", "VVT", "TTVT", "EWKT"}
L_PARTS = {"ZL", "TTL", "STL", "VVL", "TTVL", "EWKL"}
J_PARTS = {"ZJ", "TTJ", "STJ", "VVJ", "TTVJ", "EWKJ"}
SINGLE_HIGGS = {"ggH125", "qqH125", "ttH125", "VH125"}  # the 2018 fixture lists have no VH -> tautau
LUMI_2018 = "lumi_13TeV_15161718_l"


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
    sums = list(embedding_variation_sums("tt", "2018")) if embedded(embedding, "tt") else []
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
    assert {"ZTT", "TTT", "STT", "VVT"} <= lnn[(LUMI_2018, None)] and "EMB" not in lnn[(LUMI_2018, None)]
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
    assert tau.channel("mt").process("ZL").selection.weights["dy_correction"] == dy_njets_btag_weight("2018")
    assert not corrected(analysis_for("mc"))
    fake_factors = ff_measurement.build(config(analysis="bbtautau_shapesmith.ff_measurement:build", sample_lists=["sm2018_binned_v3"], dy_correction="njets_btag"))
    assert corrected(fake_factors) == {(name, p) for name in TAU_CHANNELS for p in ("ZTT", "ZL", "ZJ")}
    controls = dilepton.build(config(["em", "mm", "ee"], "bbtautau_shapesmith.dilepton:build", dy_correction="njets_btag"))
    assert corrected(controls) == {(name, "DY") for name in ("em", "mm", "ee")}


def test_diagnostics_estimate_qcd_in_the_control_regions():
    analysis = analysis_for("mc", ["et", "mt"], control_regions=True, diagnostics=True)
    mt, tt = analysis.channel("mt"), analysis.channel("tt")
    every_mc = mt.estimators[0].subtract  # as the nominal ABCD
    assert DataMinus("QCD", "btag1_ss_fail", every_mc, clip_negative=True, target="btag1_os_fail") in mt.estimators
    assert DataMinus("QCD", "lepton_antiiso_ss", every_mc, clip_negative=True, target="lepton_antiiso") in mt.estimators
    assert len([e for e in mt.estimators if isinstance(e, DataMinus)]) == 9 and mt.backgrounds().count("QCD") == 1
    abcd = [e for e in tt.estimators if isinstance(e, ABCD)]
    assert ABCD("QCD", "btag2_ss_pass", "btag2_os_fail", "btag2_ss_fail", tt.estimators[0].subtract, target="btag2_os_pass") in abcd
    assert [e.target for e in abcd] == ["nominal", "btag0_os_pass", "btag1_os_pass", "btag2_os_pass", "btag3p_os_pass"]
    for name, channel in analysis.channels.items():
        assert set(DIAGNOSTIC_VARIABLES) <= set(channel.variables) & set(analysis.style.axis_labels[name])
        for sample in channel.samples:
            available = (embedding_branches(name) if sample.kind == "embedding" else branches(name) | {"npartons"}) | V6_COLUMNS
            assert not needed_columns(channel, sample) - available, sample.nick
    assert not set(DIAGNOSTIC_VARIABLES) & set(analysis_for("mc", control_regions=True).channel("mt").variables)



def test_the_fastmtt_friend_adds_the_di_tau_variables_of_the_tau_channels():
    """fastmtt_friend: m, pt, eta and phi_fastmtt in et, mt and tt, read from every kind (the friend of data, MC and
    the embedding); em (no FastMTT friend) and a run without the switch do not need them."""
    analysis = build(config(("et", "mt", "tt", "em"), jet_fakes="ff", embedding=["et", "mt"], fit_categories="control", fastmtt_friend=True, sample_lists=["sm2018_binned_v3", EMBEDDING_LIST]))
    for name, channel in analysis.channels.items():
        tau = name in TAU_CHANNELS
        assert set(FASTMTT_VARIABLES) & set(channel.variables) == (set(FASTMTT_VARIABLES) if tau else set())
        assert set(FASTMTT_VARIABLES) <= set(analysis.style.axis_labels[name])
        for sample in channel.samples:
            assert (set(FASTMTT_VARIABLES) <= needed_columns(channel, sample)) == tau, sample.nick
    assert {v: analysis.channel("mt").variables[v].edges[::10] for v in ("m_fastmtt", "pt_fastmtt")} == {"m_fastmtt": (0.0, 100.0, 200.0, 300.0, 400.0), "pt_fastmtt": (0.0, 100.0, 200.0, 300.0)}
    assert not set(FASTMTT_VARIABLES) & set(analysis_for("ff", ["et", "mt"]).channel("mt").variables)


def test_em_is_the_ttbar_control_channel():
    analysis = build(config(("et", "mt", "tt", "em"), jet_fakes="ff", embedding=["et", "mt"], fit_categories="control", sample_lists=["sm2018_binned_v3", EMBEDDING_LIST]))
    validate(analysis)
    assert analysis.control_channels == ("em",)
    em = analysis.channel("em")
    (category,) = em.categories
    assert (category.name, category.cut, category.variable.name) == ("em_tt", "(n_bjets >= 1) & (bpair_pt_2 > 0)", "yield")
    assert em.cuts["jets"] == "n_jets >= 2" and em.regions == () and em.estimators == ()
    assert em.variations == btag_variations("2018") + prefiring_variations()  # those of the tau channels, so that the nuisances correlate
    assert set(em.backgrounds()) == {"DY", "TT", "ST", "VV", "TTV", "EWK", "W"} | SINGLE_HIGGS and "top_pt" in em.process("TT").selection.weights
    assert {s.kind for s in em.samples} == {"data", "mc"} and all(s.nick.startswith("SingleMuon_") for s in em.samples_of("data"))
    for sample in em.samples:
        assert needed_columns(em, sample) <= branches("em") | {"npartons"}, sample.nick
    lnn = {(entry.name, entry.channels): set(entry.processes) for entry in analysis.lnn}
    assert lnn[("htt_tjXsec", None)] == {"TT", "TTL", "TTT"} and {"TT", "EMB"} & lnn[(LUMI_2018, None)] == {"TT"}  # TTT of tt
    assert "TT" in lnn[("eff_e", ("et", "em"))] & lnn[("eff_m", ("mt", "em"))] & lnn[("CMS_eff_m_trigger_2018", ("mt", "em"))]  # em triggers on the muon
    assert not {"DY", "TT", "ST"} & set(analysis.ml.processes)  # no NN labels from the control channel


def test_other_channels_are_rejected():
    with pytest.raises(AnalysisError, match="supports the channels et, mt, tt and the control channels em"):
        build(config(("mt", "mm")))


def test_ml_export_labels():
    ml = analysis_for("ff", embedding=True).ml
    assert "TTT" not in ml.processes and "data" not in ml.processes
    assert ml.label_of["EMB"] == "is_DY" and ml.label_of["jetFakes"] == "is_jetFakes" and ml.region_of == {"jetFakes": "anti_iso"}
    labels = analysis_for("mc").ml.label_of
    assert labels["EWKJ"] == labels["W"] == "is_jetFakes" and labels["EWKT"] == "is_Other" and labels["ZTT"] == "is_DY"


def test_processes_of_every_channel():
    """lnN and ML labels come from the union of the channel tables: TTT is auxiliary with embedding, a background without."""
    with_emb, without_emb = analysis_for("mc", embedding=True).channel("mt"), analysis_for("mc").channel("tt")
    roles = {p.name: p.role for p in process_union([with_emb, without_emb])}
    assert roles["EMB"] == roles["ZTT"] == roles["TTT"] == "background"
    assert [p.name for p in process_union([with_emb, without_emb])] == list(dict.fromkeys(p.name for p in with_emb.processes + without_emb.processes))


@pytest.mark.parametrize("switches,match", [
    ({"sample_list": "sm2018_binned_v2"}, "sample_list"),
    ({"production": "sm2018_binned_v1"}, "production"),
    ({"embedding": "yes"}, "embedding"),
    ({"embedding": ["xx"]}, "embedding"),
    ({"embedding": ["et", "et"]}, "listed twice"),
    ({"jet_fakes": "data"}, "jet_fakes"),
    ({"sample_lists": []}, "sample_lists"),
    ({"control_regions": True, "jet_fakes": "ff"}, "control_regions requires jet_fakes: mc"),
    ({"diagnostics": True}, "diagnostics requires control_regions: true"),
    ({"ff_type": "uncorrected", "jet_fakes": "ff"}, "ff_type"),
    ({"ff_type": "raw", "shape_systematics": False}, "ff_type raw requires jet_fakes: ff"),
    ({"ff_type": "raw", "jet_fakes": "ff"}, "ff_type raw supports nominal control plots only; set shape_systematics: false"),
    ({"study_uncertainties": True, "crown_shifts": False}, "study_uncertainties requires jet_fakes: ff and crown_shifts: false"),
    ({"study_uncertainties": True, "jet_fakes": "ff"}, "study_uncertainties requires jet_fakes: ff and crown_shifts: false"),
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
