"""The NN categories of the tau channels (NN spec 6.2-6.4, SH1-SH6): the model's prefix and classes (checked against
its manifest), the schemes thesis, K1 and K2, the data-blind or thesis binning resolved by datacards, the blinding of
the HH categories, em_tt in btag1 and btag2, the cards of a run with NN categories and the blinding of the control
variable NN_score."""
import json

import numpy as np
import pytest
from shapesmith.datacards import BLINDING, NN_RECORD, write_datacard
from shapesmith.fill import run_hist
from shapesmith.histogram import NOMINAL_VARIATION, HistKey, Histogram, HistogramSet, raw_variable
from shapesmith.model import AnalysisError, ScoreBinning
from shapesmith.nn_binning import BACKGROUND_RULES, SIGNAL_RULES, THESIS_HH_EDGES, THESIS_OTHER_EDGES
from shapesmith.skim import needed_columns

from bbtautau_shapesmith.analysis import build
from bbtautau_shapesmith.ml import ML_CLASSES, raw_count_groups
from bbtautau_shapesmith.variables import NN_FINE_BINNING
from tests.helpers import EMBEDDING_LIST, config
from tests.test_analysis import T_PARTS, analysis_for
from tests.test_controls import only, write_rows

V7B5 = ML_CLASSES["v7b5"]
LT_GROUPS = {"Top": ("TTL", "STL"), "tautau": ("EMB",), "jetFakes": ("jetFakes",)}


def nn_analysis(channels=("et", "mt", "tt"), **switches):
    """A fake-factor run with embedding in et and mt (those of the channels) and the NN friend."""
    embedding = [name for name in ("et", "mt") if name in channels]
    return build(config(channels, jet_fakes="ff", embedding=embedding, nn_friend=True, sample_lists=["sm2018_binned_v3", EMBEDDING_LIST], **switches))


def test_k1_is_the_argmax_without_btag3p():
    mt = nn_analysis().channel("mt")
    assert [(c.name, c.cut) for c in mt.categories] == [(name, f"(predicted_class == {i}) & (n_bjets < 3)") for i, name in enumerate(V7B5)]
    for category in mt.categories:
        binning = category.variable.edges
        assert category.variable.expr == "predicted_max_value" and isinstance(binning, ScoreBinning) and binning.fine == NN_FINE_BINNING
        assert len(binning.fine) == 1001 and binning.thesis is None and binning.clip == ("jetFakes",)
        signal = category.name == "HH2B2Tau"
        rules = binning.rules
        assert rules.low == pytest.approx(0.2) and rules.required == LT_GROUPS
        assert (rules.max_rel_uncertainty, rules.max_bins, rules.start, rules.n_start) == ((0.5, 15, "signal", 50) if signal else (0.1, 5, "background", 5))
        assert (rules.min_background, rules.min_raw, rules.raw_min_share, rules.negative_share) == (1.0, 1.0, 0.1, 0.01)  # rule set C'
        assert binning.blinding == (0.005 if signal else None)


def test_k2_splits_the_hh_category_by_the_b_tags():
    mt = nn_analysis(nn_categories="k2").channel("mt")
    assert [(c.name, c.cut) for c in mt.categories] == [
        ("HH2B2Tau_btag1", "(predicted_class == 0) & (n_bjets == 1)"),
        ("HH2B2Tau_btag2", "(predicted_class == 0) & (n_bjets == 2)"),
    ] + [(name, f"(predicted_class == {i}) & (n_bjets < 3)") for i, name in enumerate(V7B5) if i > 0]
    assert [c.variable.edges.blinding for c in mt.categories] == [0.005, 0.005, None, None, None, None]
    assert [c.variable.edges.rules.max_bins for c in mt.categories] == [SIGNAL_RULES.max_bins] * 2 + [BACKGROUND_RULES.max_bins] * 4


def test_the_thesis_categories_and_binning():
    mt = nn_analysis(nn_categories="thesis", nn_binning="thesis").channel("mt")
    assert [c.cut for c in mt.categories] == [f"(predicted_class == {i})" for i in range(5)]
    assert [c.variable.edges.thesis for c in mt.categories] == [THESIS_HH_EDGES] + [THESIS_OTHER_EDGES] * 4
    assert all(c.variable.edges.rules is None for c in mt.categories) and mt.categories[0].variable.edges.blinding == 0.005
    assert THESIS_OTHER_EDGES == tuple(round(0.05 * i, 2) for i in range(21)) and len(THESIS_HH_EDGES) == 23


def test_the_raw_count_groups_follow_the_background_model():
    """Top, tautau and jetFakes with fake factors; TT, DY and the ABCD QCD in mc mode (the thesis classes)."""
    ff = nn_analysis()
    assert raw_count_groups(ff.channel("mt").processes, "ff") == LT_GROUPS
    assert raw_count_groups(ff.channel("tt").processes, "ff") == {"Top": ("TTL", "STL"), "tautau": ("ZTT", "TTT", "STT", "VVT", "TTVT", "EWKT"), "jetFakes": ("jetFakes",)}
    assert set(ff.channel("tt").categories[0].variable.edges.rules.required["tautau"]) == T_PARTS
    mc = analysis_for("mc", ["et", "mt"], nn_friend=True)
    assert raw_count_groups(mc.channel("mt").processes, "mc") == {"TT": ("TTL", "TTJ"), "DY": ("EMB", "ZL", "ZJ"), "QCD": ("QCD",)}
    assert raw_count_groups(mc.channel("tt").processes, "mc") == {"TT": ("TTT", "TTL", "TTJ"), "DY": ("ZTT", "ZL", "ZJ"), "QCD": ("QCD",)}
    assert mc.channel("mt").categories[0].variable.edges.rules.low == pytest.approx(1 / 6)  # thesis6


def test_the_prefix_and_classes_select_one_model_of_the_friend():
    """SH1: a friend with several models; the run reads one by its prefix and names its classes in its order."""
    thesis = list(ML_CLASSES["thesis6"])
    analysis = nn_analysis(nn_prefix="nn_m0_", nn_classes=thesis)
    mt = analysis.channel("mt")
    assert [(c.name, c.cut, c.variable.expr) for c in mt.categories] == [(name, f"(nn_m0_predicted_class == {i}) & (n_bjets < 3)", "nn_m0_predicted_max_value") for i, name in enumerate(thesis)]
    assert mt.categories[0].variable.edges.rules.low == pytest.approx(1 / 6) and mt.variables["NN_score"].expr.startswith("nn_m0_predicted_max_value - 2 * ((nn_m0_predicted_class == 0) & ")
    columns = set().union(*(needed_columns(mt, sample) for sample in mt.samples))
    assert {"nn_m0_predicted_class", "nn_m0_predicted_max_value"} <= columns and not {"predicted_class", "predicted_max_value"} & columns
    assert analysis.ml is None  # the export classes stay those of ml_labels


def test_the_skims_keep_the_columns_of_every_model_of_the_friend():
    """nn_friend_prefixes: the runs of the models of one friend production share their skims."""
    models = ["nn_m0_", "nn_m0k4_"]
    for prefix in models:
        mt = nn_analysis(nn_prefix=prefix, nn_friend_prefixes=models).channel("mt")
        assert set(mt.keep_columns) >= {f"{p}{c}" for p in models for c in ("predicted_class", "predicted_max_value")}
        columns = [needed_columns(mt, sample) for sample in mt.samples]
        assert all({"nn_m0k4_predicted_class", "nn_m0_predicted_max_value"} <= c for c in columns)
    assert nn_analysis(nn_prefix="nn_m0_").channel("mt").keep_columns == ("event", "run", "lumi")  # the run's model only
    control = build(config(("mt",), jet_fakes="ff", embedding=["mt"], fit_categories="control", nn_prefix="nn_m0_", nn_friend_prefixes=models, sample_lists=["sm2018_binned_v3", EMBEDDING_LIST]))
    assert control.channel("mt").keep_columns == ("event", "run", "lumi")  # no NN friend, no NN columns


def _manifest(tmp_path, channel, classes):
    path = tmp_path / "m0" / "2018" / channel / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"inputs": ["pt_1"], "classes": list(classes), "folds": 2, "files": {"0": "model_k2_r0.onnx", "1": "model_k2_r1.onnx"}}))
    return str(tmp_path / "m0" / "2018" / "{channel}" / "manifest.json")


def test_the_manifest_must_list_the_classes_of_the_run(tmp_path):
    for channel in ("et", "mt", "tt"):
        manifest = _manifest(tmp_path, channel, V7B5)
    assert [c.name for c in nn_analysis(nn_manifest=manifest).channel("tt").categories] == list(V7B5)
    _manifest(tmp_path, "tt", ("HH2B2Tau", "Top", "tautau", "jetFakes", "Other"))  # another order in one channel
    with pytest.raises(AnalysisError, match=r"tt/manifest.json lists the classes \['HH2B2Tau', 'Top'"):
        nn_analysis(nn_manifest=manifest)
    assert nn_analysis(("et", "mt"), nn_manifest=manifest).channel("mt").categories  # tt is not a channel of the run
    with pytest.raises(AnalysisError, match="does not exist"):
        nn_analysis(nn_manifest=str(tmp_path / "missing" / "{channel}.json"))
    custom = ["HH2B2Tau", "rest"]
    _manifest(tmp_path, "mt", custom)
    assert [c.name for c in nn_analysis(("mt",), nn_classes=custom, nn_manifest=manifest).channel("mt").categories] == custom


@pytest.mark.parametrize("switches,match", [
    ({"nn_classes": ["tautau", "Top"]}, "contain HH2B2Tau"),
    ({"nn_classes": ["HH2B2Tau", "Top", "Top"]}, "distinct"),
    ({"nn_prefix": "nn-m0"}, "nn_prefix"),
    ({"nn_categories": "k3"}, "nn_categories"),
    ({"nn_binning": "equal"}, "nn_binning"),
    ({"nn_prefix": "nn_m0_", "nn_friend_prefixes": ["nn_m0k4_"]}, "contain nn_prefix 'nn_m0_'"),
    ({"nn_prefix": "nn_m0_", "nn_friend_prefixes": ["nn_m0_", "nn_m0_"]}, "nn_friend_prefixes must be distinct"),
    ({"nn_friend_prefixes": ["nn-m0"]}, "nn_friend_prefixes"),
])
def test_invalid_nn_switches_are_reported(switches, match):
    with pytest.raises(AnalysisError, match=match):
        nn_analysis(**switches)


def test_em_tt_is_split_into_btag1_and_btag2_with_nn_categories():
    em = nn_analysis(("et", "mt", "tt", "em")).channel("em")
    assert [(c.name, c.cut, c.variable.name) for c in em.categories] == [
        ("em_tt_btag1", "(n_bjets >= 1) & (bpair_pt_2 > 0) & (n_bjets == 1)", "yield"),
        ("em_tt_btag2", "(n_bjets >= 1) & (bpair_pt_2 > 0) & (n_bjets == 2)", "yield"),
    ]
    control = build(config(("et", "mt", "tt", "em"), jet_fakes="ff", embedding=["et", "mt"], fit_categories="control", sample_lists=["sm2018_binned_v3", EMBEDDING_LIST]))
    assert [c.name for c in control.channel("em").categories] == ["em_tt"]


def _fill(hset, analysis, channel, category, variable, edges, peaked, rng):
    """Synthetic nominal histograms and unweighted counts of every card process and the data of one category: falling
    backgrounds, the signal peaked at 1 (peaked) or flat and small."""
    definition = analysis.channel(channel)
    n = len(edges) - 1
    x = 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))
    for process in (analysis.signal, *definition.backgrounds(), definition.data()):
        if process == analysis.signal:
            values = 2.0 * x**8 / np.sum(x**8) if peaked else np.full(n, 1e-3)
        elif process == definition.data():
            values = rng.poisson(40.0 * np.exp(-3.0 * x)).astype(float)
        else:
            values = 2.0 * np.exp(-3.0 * x) * (1.0 + 0.5 * rng.random(n))
        hset[HistKey(channel, category, process, "nominal", NOMINAL_VARIATION, variable)] = Histogram(edges, values, values * 0.05)
        hset[HistKey(channel, category, process, "nominal", NOMINAL_VARIATION, raw_variable(variable))] = Histogram(edges, np.full(n, 20.0), np.full(n, 20.0))


@pytest.mark.parametrize("scheme,bins,masked", [
    ("k1", ["htt_mt_1lo_2018", "htt_mt_1hi_2018", "htt_mt_2_2018", "htt_mt_3_2018", "htt_mt_4_2018", "htt_mt_5_2018"], ["htt_mt_1hi_2018"]),
    ("k2", ["htt_mt_1lo_2018", "htt_mt_1hi_2018", "htt_mt_2lo_2018", "htt_mt_2hi_2018", "htt_mt_3_2018", "htt_mt_4_2018", "htt_mt_5_2018", "htt_mt_6_2018"], ["htt_mt_1hi_2018", "htt_mt_2hi_2018"]),
])
def test_the_cards_of_a_run_with_nn_categories(tmp_path, scheme, bins, masked):
    analysis = nn_analysis(("mt", "em"), nn_categories=scheme)
    hset, rng = HistogramSet(), np.random.default_rng(5)
    for category in analysis.channel("mt").categories:
        _fill(hset, analysis, "mt", category.name, "NN_score", list(NN_FINE_BINNING), category.name.startswith("HH2B2Tau"), rng)
    for category in analysis.channel("em").categories:
        _fill(hset, analysis, "em", category.name, "yield", [0.5, 1.5], False, rng)
    card = write_datacard(hset, analysis, ["mt", "em"], tmp_path / "all")
    assert next(line for line in card.read_text().splitlines() if line.startswith("bin ")).split()[1:] == bins + ["htt_em_1_2018", "htt_em_2_2018"]
    blinding = json.loads((card.parent / BLINDING).read_text())
    assert blinding["masked"] == masked and blinding["edge_rule"] == "S/B(mu=1) > 0.005"
    assert all(blinding["categories"][hi]["lo"] == hi.replace("hi_", "lo_") for hi in masked)
    record = json.loads((card.parent / NN_RECORD).read_text())
    assert set(record) == {"mt"} and all(entry["method"] == "data_blind" for entry in record["mt"].values())
    assert all(len(entry["edges"]) - 1 <= (15 if name.startswith("HH2B2Tau") else 5) for name, entry in record["mt"].items())


# an mt event of the nominal region (2018), and what moves it into another region
NOMINAL_MT = dict(q_1=-1, q_2=1, n_bjets=1, n_jets=2, bpair_pt_2=30, pt_1=40, pt_2=35, eta_1=0.5, eta_2=0.5, mt_1=30, iso_1=0.05, m_vis=60,
                  extraelec_veto=0, extramuon_veto=0, dilepton_veto=0, hem_veto=0, id_tau_vsJet_Medium_2=1, id_tau_vsJet_VVVLoose_2=1)
SAME_SIGN, ANTI_ISO, LEPTON_ANTI_ISO, BTAG0 = {"q_1": 1}, {"id_tau_vsJet_Medium_2": 0}, {"iso_1": 0.3}, {"n_bjets": 0}
# (moved by, predicted class, score); class 0 is HH2B2Tau
NN_EVENTS = [({}, 0, 0.91), ({}, 2, 0.62), (SAME_SIGN, 0, 0.83), (SAME_SIGN, 3, 0.46), (ANTI_ISO, 0, 0.74), (ANTI_ISO, 1, 0.55),
             (LEPTON_ANTI_ISO, 0, 0.87), (BTAG0, 0, 0.93)]


@pytest.mark.parametrize("jet_fakes,kept", [
    ("ff", {"nominal": [0.62], "same_sign": [0.46, 0.83], "anti_iso": [0.55]}),
    ("mc", {"nominal": [0.62], "abcd_anti_iso": [0.55], "abcd_same_sign": [0.46, 0.83], "lepton_antiiso": [0.87], "btag1_os_pass": [0.62],
            "btag0_os_pass": [0.93]}),
])
def test_the_control_variable_nn_score_leaves_out_the_signal_category(tmp_path, jet_fakes, kept):
    """Spec 6.4 (2026-10-25, only the signal category is blinded): NN_score has no event predicted as HH2B2Tau in the
    nominal region nor in the anti-isolated region that is its jet-fake estimate, also where a control region overlaps
    them (btag1_os_pass); same sign, lepton anti-isolation and btag0 keep every event, and m_vis every event of a region."""
    switches = {"embedding": ["mt"], "sample_lists": ["sm2018_binned_v3", EMBEDDING_LIST]} if jet_fakes == "ff" else {"control_regions": True, "sample_lists": ["sm2018_binned_v3"]}
    cfg = config(["mt"], jet_fakes=jet_fakes, nn_friend=True, **switches).model_copy(update={"skim_dir": tmp_path / "skim", "output_dir": tmp_path / "out", "workers": 1})
    analysis = only(build(cfg), "mt", "data")
    channel = analysis.channel("mt")
    write_rows(cfg, channel, lambda sample, row: [row | NOMINAL_MT | moved | {"predicted_class": cls, "predicted_max_value": score} for moved, cls, score in NN_EVENTS])
    hists = run_hist(cfg, analysis, ["mt"], control=True, variables=["NN_score", "m_vis"], systematics=False, processes=None,
                     output=cfg.output_dir / "controls.root", regions=list(kept))

    for region, expected in kept.items():
        filled = hists[HistKey("mt", "inclusive", "data", region, NOMINAL_VARIATION, "NN_score")]
        assert filled.values.tolist() == Histogram.fill(filled.edges, expected).values.tolist(), region
    every = {region: hists[HistKey("mt", "inclusive", "data", region, NOMINAL_VARIATION, "m_vis")].sum() for region in kept}
    assert every["nominal"] == 2 and (every.get("same_sign") or every.get("abcd_same_sign")) == 2
