"""The validation of the NN outputs (NN spec 7.1, NV4 and NV4b; nn_outputs.py): the control variables of the outputs
and their blinding, the MC-only copies of ff_transfer, the same-sign jet fakes of ff_same_sign and the fractions of
the fake-factor payload as expressions."""
import gzip
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
from shapesmith.config import FriendConfig, load_analysis, load_config
from shapesmith.expressions import evaluate
from shapesmith.fill import run_hist
from shapesmith.histogram import NOMINAL_VARIATION, HistKey, Histogram, HistogramSet
from shapesmith.model import AnalysisError, DataMinus

from bbtautau_shapesmith import nn_outputs
from bbtautau_shapesmith.analysis import build
from bbtautau_shapesmith.constants import TAU_CHANNELS
from bbtautau_shapesmith.ml import ML_CLASSES
from bbtautau_shapesmith.weights import fake_factor_weight
from tests.helpers import DATABASE, EMBEDDING_LIST, ERAS, REPO, config
from tests.test_controls import only, write_rows
from tests.test_nn_categories import ANTI_ISO, BTAG0, LEPTON_ANTI_ISO, NOMINAL_MT, SAME_SIGN

V7B5 = ML_CLASSES["v7b5"]
CROWN_PAYLOADS = REPO.parent.parent / "KingMaker" / "CROWN" / "analysis_configurations" / "bbtautau" / "payloads" / "fake_factors" / "sm" / "fake-factors-2026-10-19"


def write_blinding(path, edges: dict[str, float], era="2018"):
    """A blinding.json of datacards with the HH bins of every channel (bin index -> edge)."""
    categories = {f"htt_{ch}_{index}hi_{era}": {"lo": f"htt_{ch}_{index}lo_{era}", "edge": edge, "s_over_b_first_hi_bin": 0.01}
                  for ch in TAU_CHANNELS for index, edge in edges.items()}
    path.write_text(json.dumps({"edge_rule": "S/B(mu=1) > 0.005", "masked": list(categories), "categories": categories, "asimov_data": True}))
    return path


def ff_run(channels=("mt",), **switches):
    """A fake-factor run with embedding in et and mt and the NN friend (unless the switches say otherwise)."""
    embedding = [name for name in ("et", "mt") if name in channels]
    return config(channels, **{"jet_fakes": "ff", "embedding": embedding, "nn_friend": True, "sample_lists": ["sm2018_binned_v3", EMBEDDING_LIST], **switches})


def test_the_output_variables_of_k1_and_k2(tmp_path):
    blinding = write_blinding(tmp_path / "blinding.json", {"1": 0.9, "2": 0.8})
    mt = build(ff_run(nn_outputs=True, nn_blinding=str(blinding), nn_prefix="nn_m0k4_")).channel("mt")
    names = set(mt.variables)
    assert {f"NN_cat_{c}" for c in V7B5} | {f"NN_P_{c}" for c in V7B5} | {"NN_ar_HH2B2Tau"} <= names
    assert {f"NN_{x}_{c}" for x in nn_outputs.INPUTS for c in V7B5} <= names
    assert not any(name.startswith("NNmc_") for name in names)  # only with ff_transfer
    assert names - {name for name in names if name.startswith("NN_")} == {"yield"}  # the outputs replace the control variables
    hh, top = mt.variables["NN_cat_HH2B2Tau"], mt.variables["NN_cat_Top"]
    assert hh.expr.startswith("nn_m0k4_predicted_max_value - 1000 * (~((nn_m0k4_predicted_class == 0) & (n_bjets < 3)) | ((")
    assert "(nn_m0k4_predicted_max_value >= 0.9))" in hh.expr and top.expr == "nn_m0k4_predicted_max_value - 1000 * (~((nn_m0k4_predicted_class == 2) & (n_bjets < 3)))"
    assert mt.variables["NN_P_tautau"].expr.startswith("nn_m0k4_nn_score_tautau - 2 * ((nn_m0k4_predicted_class == 0) & ")
    assert len(hh.edges) == 1001 and mt.variables["NN_n_jets_Top"].edges[0] == -0.5
    k2 = build(ff_run(nn_outputs=True, nn_blinding=str(blinding), nn_categories="k2")).channel("mt")
    assert "(predicted_max_value >= 0.9))" in k2.variables["NN_cat_HH2B2Tau_btag1"].expr and "(predicted_max_value >= 0.8))" in k2.variables["NN_cat_HH2B2Tau_btag2"].expr
    assert {"NN_ar_HH2B2Tau_btag1", "NN_ar_HH2B2Tau_btag2"} <= set(k2.variables)


def test_the_blinding_edges_of_a_channel(tmp_path):
    path = write_blinding(tmp_path / "blinding.json", {"1": 0.892}, era="2017")
    assert nn_outputs.blinding_edges(path, "mt", "2017") == {"1": 0.892}
    with pytest.raises(AnalysisError, match="no blinded HH category of mt 2018"):
        nn_outputs.blinding_edges(path, "mt", "2018")
    with pytest.raises(AnalysisError, match="no blinding edge for the HH category HH2B2Tau_btag2"):
        build(ff_run(nn_outputs=True, nn_blinding=str(write_blinding(tmp_path / "k1.json", {"1": 0.9})), nn_categories="k2"))


@pytest.mark.parametrize("switches,match", [
    ({"nn_outputs": True, "nn_friend": False, "fit_categories": "none"}, "nn_outputs requires nn_friend"),
    ({"nn_blinding": "x.json"}, "nn_blinding requires nn_outputs"),
    ({"nn_ff_payload": "x.json.gz"}, "nn_ff_payload requires nn_outputs"),
])
def test_invalid_output_switches_are_reported(tmp_path, switches, match):
    with pytest.raises(AnalysisError, match=match):
        build(ff_run(**switches))
    with pytest.raises(AnalysisError, match="ff_same_sign requires jet_fakes: ff"):
        build(config(["mt"], ff_same_sign=True))


# (moved by, predicted class, score); class 0 is HH2B2Tau, the blinding edge 0.85
EVENTS = [({}, 0, 0.91), ({}, 0, 0.62), ({}, 0, 0.85), ({}, 2, 0.55), (SAME_SIGN, 0, 0.83), (SAME_SIGN, 0, 0.95), (ANTI_ISO, 0, 0.74), (ANTI_ISO, 0, 0.88),
          (ANTI_ISO, 1, 0.45), (SAME_SIGN | ANTI_ISO, 0, 0.97), (LEPTON_ANTI_ISO, 0, 0.87), (BTAG0, 0, 0.93)]
EDGE = 0.85


def assert_filled(hists, region, variable, expected, process="data"):
    """The histogram holds exactly the values `expected`, one event each."""
    h = hists[HistKey("mt", "inclusive", process, region, NOMINAL_VARIATION, variable)]
    assert h.values.tolist() == Histogram.fill(h.edges, expected).values.tolist(), (region, variable)


@pytest.mark.parametrize("edges", [True, False])
def test_the_hh_category_shows_its_lo_part_only_in_the_nominal_region(tmp_path, edges):
    """Spec 6.4: NN_cat_HH2B2Tau has no data event above the blinding edge in the nominal region nor in the
    anti-isolated region of its jet fakes (without nn_blinding none at all there); the same-sign region (and its
    anti-isolated part) keeps every event; NN_ar_ holds the anti-isolated events only, every score; NN_P_ leaves out
    the events NN_score leaves out."""
    blinding = write_blinding(tmp_path / "blinding.json", {"1": EDGE})
    cfg = ff_run(nn_outputs=True, ff_same_sign=True, **({"nn_blinding": str(blinding)} if edges else {})).model_copy(
        update={"skim_dir": tmp_path / "skim", "output_dir": tmp_path / "out", "workers": 1})
    analysis = only(build(cfg), "mt", "data")
    channel = analysis.channel("mt")
    rows = [NOMINAL_MT | moved | {"predicted_class": cls, "predicted_max_value": score, **{f"nn_score_{c}": (score if i == cls else (1 - score) / 4) for i, c in enumerate(V7B5)}}
            for moved, cls, score in EVENTS]
    write_rows(cfg, channel, lambda sample, row: [row | r for r in rows])
    regions = ["nominal", "anti_iso", "same_sign", "same_sign_anti_iso"]
    variables = ["NN_cat_HH2B2Tau", "NN_ar_HH2B2Tau", "NN_P_HH2B2Tau", "NN_cat_tautau", "NN_n_jets_HH2B2Tau"]
    hists = run_hist(cfg, analysis, ["mt"], control=True, variables=variables, systematics=False, processes=None, output=cfg.output_dir / "c.root", regions=regions)

    lo = [0.62] if edges else []
    assert_filled(hists, "nominal", "NN_cat_HH2B2Tau", lo)
    assert_filled(hists, "anti_iso", "NN_cat_HH2B2Tau", [0.74] if edges else [])
    assert_filled(hists, "same_sign", "NN_cat_HH2B2Tau", [0.83, 0.95])
    assert_filled(hists, "same_sign_anti_iso", "NN_cat_HH2B2Tau", [0.97])
    assert_filled(hists, "nominal", "NN_ar_HH2B2Tau", [])
    assert_filled(hists, "same_sign", "NN_ar_HH2B2Tau", [])
    assert_filled(hists, "anti_iso", "NN_ar_HH2B2Tau", [0.74, 0.88])
    assert_filled(hists, "nominal", "NN_P_HH2B2Tau", [(1 - 0.55) / 4])  # the HH-predicted events left out
    assert_filled(hists, "nominal", "NN_cat_tautau", [])
    assert_filled(hists, "anti_iso", "NN_cat_tautau", [0.45])
    assert_filled(hists, "nominal", "NN_n_jets_HH2B2Tau", [2.0] * len(lo))


def test_the_mc_only_copies_hold_no_data(tmp_path):
    """With ff_transfer: NNmc_ variables hold every simulated event of the HH category (no blinding needed) and no
    data; the anti-isolated region with the fake factor carries the decay-mode non-closure of the fake factors."""
    cfg = config(["mt"], jet_fakes="mc", nn_friend=True, fit_categories="none", nn_classes=list(V7B5), control_regions=True, ff_transfer=True, nn_outputs=True,
                 sample_lists=["sm2018_binned_v3"]).model_copy(
        update={"skim_dir": tmp_path / "skim", "output_dir": tmp_path / "out", "workers": 1})
    analysis = only(build(cfg), "mt", "data", "TTJ")
    channel = analysis.channel("mt")
    assert {"NNmc_cat_HH2B2Tau", "NNmc_n_jets_HH2B2Tau", "NNmc_mass_2_HH2B2Tau"} | {f"NNmc_P_{c}" for c in V7B5} <= set(channel.variables)
    assert any(v.name.startswith("CMS_ff_dm_nonclosure_mt") for v in build(cfg).channel("mt").variations)

    def rows(sample, row):
        extra = {"gen_match_1": 4, "gen_match_2": 6, "fake_factor": 0.1} if sample.kind == "mc" else {"fake_factor": 0.1}
        return [row | NOMINAL_MT | extra | {"predicted_class": 0, "predicted_max_value": s, **{f"nn_score_{c}": s for c in V7B5}} for s in (0.62, 0.97)]

    write_rows(cfg, channel, rows)
    hists = run_hist(cfg, analysis, ["mt"], control=True, variables=["NNmc_cat_HH2B2Tau", "NN_cat_HH2B2Tau", "NNmc_P_Top"], systematics=False, processes=None,
                     output=cfg.output_dir / "c.root", regions=["nominal"])
    assert_filled(hists, "nominal", "NNmc_cat_HH2B2Tau", [])
    assert_filled(hists, "nominal", "NNmc_P_Top", [])
    simulated = hists[HistKey("mt", "inclusive", "TTJ", "nominal", NOMINAL_VARIATION, "NNmc_cat_HH2B2Tau")]
    assert (simulated.values / simulated.values.max()).tolist() == Histogram.fill(simulated.edges, [0.62, 0.97]).values.tolist()
    assert_filled(hists, "nominal", "NN_cat_HH2B2Tau", [])  # without nn_blinding nothing of HH in the nominal region


def test_the_mc_only_copies_leave_out_data_and_embedding():
    cfg = config(["mt"], jet_fakes="mc", nn_friend=True, fit_categories="none", nn_classes=list(V7B5), control_regions=True, ff_transfer=True, nn_outputs=True,
                 sample_lists=["sm2018_binned_v3"])
    variables = build(cfg).channel("mt").variables
    frame = pd.DataFrame({"predicted_class": [0, 0, 0], "predicted_max_value": [0.9, 0.9, 0.9], "n_bjets": [1, 1, 1], "n_jets": [2, 2, 2], "mass_2": [0.5] * 3,
                          "nn_score_Top": [0.3] * 3, "is_mc": [True, False, False], "is_data": [False, True, False], "is_embedding": [False, False, True]})
    for name in ("NNmc_cat_HH2B2Tau", "NNmc_P_Top", "NNmc_n_jets_HH2B2Tau"):
        assert (evaluate(frame, variables[name].expr) >= 0).tolist() == [True, False, False], name


def test_the_same_sign_jet_fakes_in_ff_mode():
    on, off = build(ff_run(ff_same_sign=True)), build(ff_run())
    region = on.channel("mt").region("same_sign_anti_iso")
    assert region.replace_cuts["os"] == "((q_1 * q_2) > 0)" and region.add_weights == {"fake_factor": fake_factor_weight("mt")}
    added = [e for e in on.channel("mt").estimators if e not in off.channel("mt").estimators]
    assert added == [DataMinus("jetFakes", "same_sign_anti_iso", next(e for e in off.channel("mt").estimators if isinstance(e, DataMinus)).subtract, target="same_sign")]
    assert on.channel("mt").variables["NN_score"].expr == off.channel("mt").variables["NN_score"].expr  # the same-sign region stays unblinded


# -- NV4b: the fractions of the payload as expressions ---------------------------------------------------------------


def _binning(column, edges, values):
    return {"nodetype": "binning", "input": column, "edges": edges, "flow": "clamp", "content": values}


def write_payload(path, channel):
    """A fake-factor payload with the fraction corrections of the channel (the layout of ShapeSmith's measurement)."""
    column = "mt_1" if channel != "tt" else "m_vis"
    names = ["process_fractions"] + (["process_fractions_subleading"] if channel == "tt" else [])
    corrections = []
    for leg, name in enumerate(names):
        def by_process(shift):
            ttbar = [_binning(column, [0.0, 30.0, 70.0], [0.6 + shift + leg / 10, 0.4 + shift]), _binning(column, [0.0, 40.0, 55.0, 70.0], [0.7, 0.5 + shift, 0.3])]
            qcd = [_binning(column, [0.0, 30.0, 70.0], [0.4, 0.6]), _binning(column, [0.0, 40.0, 55.0, 70.0], [0.3, 0.5, 0.7])]
            return {"nodetype": "category", "input": "process", "content": [
                {"key": "QCD", "value": {"nodetype": "binning", "input": "n_jets", "edges": [-0.5, 2.5, 22.5], "flow": "clamp", "content": qcd}},
                {"key": "ttbar", "value": {"nodetype": "binning", "input": "n_jets", "edges": [-0.5, 2.5, 22.5], "flow": "clamp", "content": ttbar}}]}
        data = {"nodetype": "category", "input": "syst", "default": by_process(0.0),
                "content": [{"key": f"{name}nominal", "value": by_process(0.0)}, {"key": f"{name}frac_ttbar_J_up", "value": by_process(0.05)}]}
        corrections.append({"name": name, "version": 1, "inputs": [{"name": "process", "type": "string"}, {"name": column, "type": "real"},
                                                                  {"name": "n_jets", "type": "real"}, {"name": "syst", "type": "string"}],
                            "output": {"name": "fraction", "type": "real"}, "data": data})
    with gzip.open(path, "wt") as handle:
        json.dump({"schema_version": 2, "corrections": corrections}, handle)
    return path


@pytest.mark.parametrize("channel", ["mt", "tt"])
@pytest.mark.parametrize("syst", ["nominal", "frac_ttbar_J_up"])
def test_the_payload_fraction_as_an_expression_equals_correctionlib(tmp_path, channel, syst):
    correctionlib = pytest.importorskip("correctionlib")
    path = write_payload(tmp_path / f"fake_factors_{channel}.json.gz", channel)
    rng = np.random.default_rng(1)
    column = "mt_1" if channel != "tt" else "m_vis"
    frame = pd.DataFrame({column: rng.uniform(-10, 90, 500), "n_jets": rng.integers(0, 6, 500).astype(float),
                          "id_tau_vsJet_Medium_1": rng.integers(0, 2, 500), "id_tau_vsJet_Medium_2": rng.integers(0, 2, 500)})
    frame.loc[:3, column] = [30.0, 40.0, 55.0, 70.0]  # the edges
    if channel == "tt":  # one leg anti-isolated, as in the anti-isolated region
        frame["id_tau_vsJet_Medium_2"] = 1 - frame["id_tau_vsJet_Medium_1"]
    got = evaluate(frame, nn_outputs.fraction_expression(path, channel, syst=syst))
    cset = correctionlib.CorrectionSet.from_file(str(path))
    names = ["process_fractions"] if channel != "tt" else ["process_fractions", "process_fractions_subleading"]
    legs = {name: cset[name].evaluate("ttbar", frame[column].to_numpy(), frame["n_jets"].to_numpy(), f"{name}{syst}") for name in names}
    expected = legs["process_fractions"] if channel != "tt" else np.where(frame["id_tau_vsJet_Medium_1"] < 0.5, legs["process_fractions"], legs["process_fractions_subleading"])
    np.testing.assert_allclose(got, expected, rtol=1e-15, atol=0)


@pytest.mark.parametrize("channel", ["et", "mt", "tt"])
def test_the_fraction_regions_of_the_measured_payload(channel):
    """On the payload of ff_v7b (CROWN bbtautau, when the checkout is there): the expressions equal correctionlib."""
    correctionlib = pytest.importorskip("correctionlib")
    path = CROWN_PAYLOADS / "2018" / f"fake_factors_{channel}.json.gz"
    if not path.exists():
        pytest.skip(f"no CROWN payload {path}")
    regions = nn_outputs.fraction_regions(path, channel)
    assert [r.name for r in regions] == ["anti_iso_ttbar_fraction", "anti_iso_ttbar_fraction_up"]
    rng = np.random.default_rng(2)
    column = "mt_1" if channel != "tt" else "m_vis"
    frame = pd.DataFrame({column: rng.uniform(0, 250, 300), "n_jets": rng.integers(2, 8, 300).astype(float),
                          "id_tau_vsJet_Medium_1": np.ones(300), "id_tau_vsJet_Medium_2": np.zeros(300)})
    cset = correctionlib.CorrectionSet.from_file(str(path))
    for region, syst in zip(regions, ("nominal", "frac_ttbar_J_up")):
        name = "process_fractions_subleading" if channel == "tt" else "process_fractions"  # the tau 2 fails in these rows
        expected = cset[name].evaluate("ttbar", frame[column].to_numpy(), frame["n_jets"].to_numpy(), f"{name}{syst}")
        # numexpr parses the decimal literals itself, to 1 ulp
        np.testing.assert_allclose(evaluate(frame, region.add_weights["ttbar_fraction"]), expected, rtol=1e-15, atol=0)
        assert region.replace_cuts == {"tau_iso": nn_outputs.anti_iso_cut(channel)}


def test_unsupported_payload_nodes_are_refused():
    with pytest.raises(AnalysisError, match="flow 'error'|not supported"):
        nn_outputs.node_expression({"nodetype": "binning", "input": "x", "edges": [0, 1], "flow": {"nodetype": "formula"}, "content": [1.0]}, {})
    with pytest.raises(AnalysisError, match="no content for process"):
        nn_outputs.node_expression({"nodetype": "category", "input": "process", "content": [{"key": "QCD", "value": 1.0}]}, {"process": "ttbar"})
    assert nn_outputs.node_expression({"nodetype": "binning", "input": "x", "edges": [0.0, 1.0, 2.0], "flow": "error", "content": [1.0, 2.0]}, {}) == \
        "(((x >= 0.0) & (x < 1.0)) * 1.0 + ((x >= 1.0) & (x < 2.0)) * 2.0)"


# -- the run YAML of the mc-mode checks ------------------------------------------------------------------------------

STORE = "/store/user/sdaigler/CROWN/ntuples"
# run YAML -> (the run it copies, the switches it changes)
OUTPUT_RUNS = {"smrun2_binned_v7b_nn_m0k4_nv4_control": ("smrun2_binned_v7b_nn_validation_control", {
    "nn_friend": True, "nn_prefix": "nn_m0k4_", "nn_classes": list(V7B5), "nn_outputs": True,
    "nn_ff_payload": "/work/sdaigler/bbtautau/KingMaker/CROWN/analysis_configurations/bbtautau/payloads/fake_factors/sm/fake-factors-2026-10-19/{year}/fake_factors_{channel}.json.gz"})}


def _load(name, era, *overrides):
    return load_config(REPO / "configs" / f"{name}.yaml", [f"era={era}", f"sample_database={DATABASE}", *overrides])


@pytest.mark.parametrize("name", OUTPUT_RUNS)
@pytest.mark.parametrize("era", ERAS)
def test_the_output_run_is_the_validation_control_run_with_the_nn_outputs(tmp_path, name, era):
    copied, switches = OUTPUT_RUNS[name]
    run, base = _load(name, era), _load(copied, era)
    assert (str(run.skim_dir), str(run.output_dir)) == (f"/ceph/sdaigler/nn/nv_m0k4/skims/tau_control_nn/{era}", f"/ceph/sdaigler/nn/nv_m0k4/output/tau_control_nn/{era}")
    expected = {key: value for key, value in json.loads(base.model_dump_json()).items() if key not in ("skim_dir", "output_dir", "workers", "combine")}
    actual = {key: value for key, value in json.loads(run.model_dump_json()).items() if key not in ("skim_dir", "output_dir", "workers", "combine")}
    for name_ in ("nn_validation", "pseudo_shifts"):
        expected["switches"].pop(name_)
    expected["switches"].update(switches)
    expected["ntuples"]["friends"] += [  # the NN friend the copies of the pseudo friend claim
        FriendConfig(base=f"{STORE}/smrun2_binned_v7/CROWNFriend/nn_m0_v1", applies_to=("data",), channels=TAU_CHANNELS).model_dump(mode="json"),
        FriendConfig(base=f"{STORE}/smrun2_binned_v7b/CROWNFriend/nn_m0_v1", applies_to=("mc", "embedding"), channels=TAU_CHANNELS).model_dump(mode="json"),
    ]
    assert actual == expected and run.workers <= 4
    # it builds with a payload of the layout of the measurement
    for channel in TAU_CHANNELS:
        write_payload(tmp_path / f"fake_factors_{channel}.json.gz", channel)
    analysis = load_analysis(_load(name, era, f"switches.nn_ff_payload={tmp_path}/fake_factors_{{channel}}.json.gz"))
    for channel in TAU_CHANNELS:
        needed = {"anti_iso", "abcd_anti_iso", "anti_iso_ttbar_fraction", "anti_iso_ttbar_fraction_up"} | (set() if channel == "tt" else {"w_highmt_pass", "lepton_antiiso"})
        assert needed <= {r.name for r in analysis.channel(channel).regions}
        assert {"NNmc_cat_HH2B2Tau", "NN_ar_HH2B2Tau", "NN_cat_Top"} <= set(analysis.channel(channel).variables)


# -- NV9: the pseudo-shifts of the NN outputs (S0+) ------------------------------------------------------------------


def _pseudo_friend():
    spec = importlib.util.spec_from_file_location("pseudo_nn_friend", REPO / "scripts" / "pseudo_nn_friend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_pseudo_shifts_read_the_shifted_outputs_in_a_run_with_the_nn_friend():
    """S0+ (NV9): with the NN friend each pseudo-shift reads the friend columns of the model under the shift
    (<column>__<shift>), in the input validation (no friend) nothing changes, so its skims stay valid."""
    nn = build(ff_run(("et", "mt", "tt"), pseudo_shifts=True, nn_prefix="nn_m0k4_")).channel("mt")
    pseudo = [v for v in nn.variations if v.name.startswith("CMS_pseudo_")]
    assert {v.name for v in pseudo} == {f"CMS_pseudo_{kind}_2018{d}" for kind in ("jes", "tes", "emb_tes") for d in ("Up", "Down")}
    columns = ["nn_m0k4_predicted_class", "nn_m0k4_predicted_max_value", *(f"nn_m0k4_nn_score_{c}" for c in V7B5)]
    for v in pseudo:
        assert v.suffix == "" and {c: v.derived[c] for c in columns} == {c: f"{c}__{v.name}" for c in columns} and "met" in v.derived
    assert {v.name for v in _pseudo_friend().pseudo_shifts(nn, columns)} == {v.name for v in pseudo}
    validation = build(config(["mt"], jet_fakes="ff", embedding=["mt"], sample_lists=["sm2018_binned_v3", EMBEDDING_LIST], nn_validation=True, pseudo_shifts=True,
                              fit_categories="none")).channel("mt")
    assert not any(c.startswith("nn_") for v in validation.variations if v.name.startswith("CMS_pseudo_") for c in v.derived)
    assert _pseudo_friend().pseudo_shifts(validation, columns) == []


def test_the_pseudo_friend_scores_the_shifted_inputs_behind_the_shifted_gate(tmp_path, monkeypatch):
    friend = _pseudo_friend()
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"event": np.array([1, 2, 3], dtype=np.uint64), "pt_2": [30.0, 40.0, 50.0], "other": [7.0, 8.0, 9.0]})
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp_path / "in.parquet")

    def model_columns(model, rows, gate):  # stands in for the ONNX model: its score is pt_2 / 100 inside the gate
        return {"p_predicted_class": np.where(gate, 0, -1).astype(np.int32), "p_predicted_max_value": np.where(gate, rows["pt_2"] / 100, -1.0).astype(np.float32)}

    monkeypatch.setattr(friend, "friend_columns", model_columns)
    shift = {"name": "CMS_pseudo_tes_2018Up", "suffix": "__CMS_pseudo_tes_2018Up", "derived": {"pt_2": "(1.5 * pt_2)"}}
    job = {"source": str(tmp_path / "in.parquet"), "target": str(tmp_path / "out" / "in.parquet"), "read": {"event", "pt_2"}, "gate": ["pt_2 > 35"],
           "models": [{"files": ["model"]}], "shifts": [shift]}
    record = friend.score_file(job)
    written = pq.read_table(tmp_path / "out" / "in.parquet").to_pandas()
    assert record["events"] == 3 and record["gate"] == 2
    np.testing.assert_allclose(written["p_predicted_max_value"], [-1.0, 0.4, 0.5], rtol=1e-6)
    np.testing.assert_allclose(written["p_predicted_max_value__CMS_pseudo_tes_2018Up"], [0.45, 0.6, 0.75], rtol=1e-6)  # all three pass the shifted gate
    assert written["p_predicted_class__CMS_pseudo_tes_2018Up"].tolist() == [0, 0, 0] and written["other"].tolist() == [7.0, 8.0, 9.0]
    assert written["pt_2"].tolist() == [30.0, 40.0, 50.0]  # the skim keeps its nominal columns


# -- the scripts of NV4 (FF closure) and NV4b (fractions) ------------------------------------------------------------


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _h(values, variances=None, edges=None):
    values = np.asarray(values, dtype=float)
    return Histogram(np.arange(len(values) + 1, dtype=float) if edges is None else edges, values, values if variances is None else np.asarray(variances, dtype=float))


def mc_outputs_channel():
    return build(config(["mt"], jet_fakes="mc", nn_friend=True, fit_categories="none", nn_classes=list(V7B5), control_regions=True, ff_transfer=True, nn_outputs=True,
                        sample_lists=["sm2018_binned_v3"])).channel("mt")


def test_the_ff_closure_reads_the_mc_only_copies_where_the_data_variables_hide_events():
    transfer = _script("nn_outputs_transfer")
    names = transfer.outputs(mc_outputs_channel())
    assert names["NN_cat_HH2B2Tau"] == "NNmc_cat_HH2B2Tau" and names["NN_cat_Top"] == "NN_cat_Top"
    assert names["NN_P_tautau"] == "NNmc_P_tautau" and names["NN_mass_2_HH2B2Tau"] == "NNmc_mass_2_HH2B2Tau" and names["NN_n_jets_Other"] == "NN_n_jets_Other"
    assert sum(name.startswith("NN_cat_") for name in names) == 5 and sum(name.startswith("NN_P_") for name in names) == 5


def test_the_ff_closure_test_and_the_category_shares():
    transfer = _script("nn_outputs_transfer")
    hset = HistogramSet()
    for region, scale in (("nominal", 1.0), ("anti_iso", 2.0)):  # AR x FF twice the SR: the same shape
        for variable, values in (("NN_cat_Top", [10.0, 20.0, 30.0]), ("NN_cat_Other", [5.0, 5.0, 5.0])):
            hset[HistKey("mt", "inclusive", "TTJ", region, NOMINAL_VARIATION, variable)] = _h(np.array(values) * scale, np.array(values) * scale * scale)
    for direction, factor in (("Up", 1.1), ("Down", 0.9)):  # a shift of the Top category only
        hset[HistKey("mt", "inclusive", "TTJ", "anti_iso", f"CMS_ff_x{direction}", "NN_cat_Top")] = _h(np.array([20.0, 40.0, 60.0]) * factor)
    same = transfer.compare(transfer.summed(hset, "mt", "nominal", ["TTJ"], "NN_cat_Top"), transfer.summed(hset, "mt", "anti_iso", ["TTJ"], "NN_cat_Top"), {})
    assert same["chi2"] == pytest.approx(0.0) and same["p"] == pytest.approx(1.0)
    sr = transfer.category_yields(hset, "mt", "nominal", ["TTJ"], ["NN_cat_Top", "NN_cat_Other"])
    ar = transfer.category_yields(hset, "mt", "anti_iso", ["TTJ"], ["NN_cat_Top", "NN_cat_Other"])
    assert sr.values.tolist() == [60.0, 15.0] and ar.values.tolist() == [120.0, 30.0] and ar.variances.tolist() == [240.0, 60.0]  # sum of the variances
    shifted = transfer.category_yields(hset, "mt", "anti_iso", ["TTJ"], ["NN_cat_Top", "NN_cat_Other"], "CMS_ff_xUp")
    assert shifted.values.tolist() == [pytest.approx(132.0), 30.0]  # the Other category without the shift: its nominal
    moved = _h([60.0, 40.0])  # the SR with twice the Other share
    result = transfer.compare(moved, ar, {"CMS_ff_x": (shifted, transfer.category_yields(hset, "mt", "anti_iso", ["TTJ"], ["NN_cat_Top", "NN_cat_Other"], "CMS_ff_xDown"))})
    assert result["ndf"] == 1 and result["p"] < 0.05 and result["ratio"][1] == pytest.approx((40 / 100) / (30 / 150))


def test_the_fractions_of_a_bin_in_the_definition_of_the_payload():
    fractions = _script("nn_outputs_fractions")
    channel = mc_outputs_channel()
    analysis = type("A", (), {"channel": lambda self, name: channel})()
    hset = HistogramSet()
    edges = [0.0, 0.5, 1.0]

    def put(process, region, values):
        hset[HistKey("mt", "inclusive", process, region, NOMINAL_VARIATION, "NN_cat_Top")] = _h(values, edges=np.array([0.0, 0.25, 0.5, 0.75, 1.0]))

    # AR per fine bin: data 100, genuine 10 (ZTT), lepton fakes 5 (ZL), TTJ 30, W 20, other J 5 -> QCD 30
    for process, values in (("data", [50, 50, 60, 40]), ("ZTT", [5, 5, 5, 5]), ("ZL", [2.5, 2.5, 2.5, 2.5]), ("TTJ", [15, 15, 20, 10]), ("W", [10, 10, 5, 5]), ("ZJ", [2.5, 2.5, 2.5, 2.5])):
        put(process, "abcd_anti_iso", values)
    truth = fractions.truth(hset, analysis, "mt", "NN_cat_Top", edges)
    # bin 1: QCD = 100 - (10 + 5 + 30 + 20 + 5) = 30, f = 30 / (30 + 30); bin 2: QCD = 100 - (10 + 5 + 30 + 10 + 5) = 40
    assert truth["f_ttbar"] == pytest.approx([0.5, 30 / 70])
    assert truth["shares"]["W"] == pytest.approx([20 / 85, 10 / 85]) and truth["shares"]["QCD"] == pytest.approx([30 / 85, 40 / 85])
    # a bin whose simulation exceeds the data: QCD 0 in the payload's definition, f = 1
    put("data", "abcd_anti_iso", [50, 50, 20, 5])
    assert fractions.truth(hset, analysis, "mt", "NN_cat_Top", edges)["f_ttbar"][1] == pytest.approx(1.0)
    put("data", "abcd_anti_iso", [50, 50, 60, 40])
    # the payload: f = 0.4 on the data, 0.2 on the subtracted genuine and lepton fakes (0.5 and 0.3 up), averaged over
    # data - genuine - lepton fakes: (0.4 x 100 - 0.2 x 15) / 85
    for region, (f_data, f_sim) in (("anti_iso_ttbar_fraction", (0.4, 0.2)), ("anti_iso_ttbar_fraction_up", (0.5, 0.3))):
        for process, values, f in (("data", [50, 50, 60, 40], f_data), ("ZTT", [5, 5, 5, 5], f_sim), ("ZL", [2.5, 2.5, 2.5, 2.5], f_sim)):
            put(process, region, np.array(values) * f)
    payload = fractions.payload_fraction(hset, analysis, "mt", "NN_cat_Top", edges)
    assert payload["f_ttbar"] == pytest.approx([37 / 85] * 2) and payload["delta_f"] == pytest.approx([(45.5 - 37) / 85] * 2)
    limit = HistogramSet()
    fine = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    limit[HistKey("mt", "Top", "jetFakes", "nominal", NOMINAL_VARIATION, "NN_score")] = _h([10, 10, 10, 10], edges=fine)
    for name, up, down in (("CMS_ff_process_fractionsfrac_ttbar_J_mt_2018", 11, 9), ("CMS_ff_process_fractions_subleadingfrac_ttbar_J_mt_2018", 10.5, 9.5),
                           ("CMS_ff_QCDStatShift_mt_2018", 12, 8)):
        limit[HistKey("mt", "Top", "jetFakes", "nominal", f"{name}Up", "NN_score")] = _h([up] * 4, edges=fine)
        limit[HistKey("mt", "Top", "jetFakes", "nominal", f"{name}Down", "NN_score")] = _h([down] * 4, edges=fine)
    template = fractions.template(limit, "mt", "Top", edges)
    assert template["T"] == [20.0, 20.0] and template["T_fraction_up_minus_T"] == [3.0, 3.0] and template["ff_shifts"] == 3  # both legs' fractions
    assert template["sigma_ff"] == pytest.approx([np.sqrt(2.0**2 + 1.0**2 + 4.0**2)] * 2)
