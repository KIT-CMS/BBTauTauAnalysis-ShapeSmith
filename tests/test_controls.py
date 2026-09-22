"""Control selections tested on events, including the skim/region acceptance contract."""
import numpy as np
import pandas as pd
import pytest

from shapesmith.config import RunConfig
from shapesmith.expressions import apply_region, mask, selection_columns, weight
from shapesmith.model import AnalysisError
from shapesmith.skim import required_columns
from bbtautau_shapesmith.analysis import build as build_tau
from bbtautau_shapesmith.dilepton_analysis import build
from bbtautau_shapesmith.dilepton_selection import channel_definition
from bbtautau_shapesmith.selection import channel_definition as tau_channel
from tests.helpers import DATABASE, REPO, branches


def config(channels=("em", "mm", "ee"), **switches):
    return RunConfig(analysis="unused", era="2018", channels=channels,
                     ntuples={"base": "/unused"}, skim_dir="/unused/skim", output_dir="/unused/output",
                     sample_database=DATABASE, switches=switches)


def event(**updates):
    return dict(pt_1=40., pt_2=30., iso_1=.05, iso_2=.05, q_1=1, q_2=-1,
                extraelec_veto=0, extramuon_veto=0, m_vis=90., n_jets=2,
                n_bjets=1, trg_single_mu24=1, trg_single_ele32=1) | updates


@pytest.mark.parametrize("channel,changes,expected", [
    ("em", [{}, {"pt_2": 26}, {"pt_1": 16}, {"n_jets": 1}, {"q_2": 1}, {"iso_2": .15}],
     [True, False, True, False, False, False]),
    ("mm", [{}, {"pt_1": 26}, {"pt_2": 15}, {"m_vis": 70}, {"m_vis": 110}, {"n_jets": 0}],
     [True, False, False, False, False, True]),
    ("ee", [{}, {"pt_1": 34}, {"pt_2": 16}, {"trg_single_ele32": 0}, {"iso_1": .15}],
     [True, False, True, False, False]),
])
def test_dilepton_selection_boundaries(channel, changes, expected):
    frame = pd.DataFrame([event(**change) for change in changes])
    assert mask(frame, channel_definition(channel).baseline.cuts).tolist() == expected


@pytest.mark.parametrize("channel", ["em", "mm", "ee"])
def test_btag_regions_partition_two_jet_events(channel):
    frame = pd.DataFrame([event(n_bjets=n, n_jets=max(2, n)) for n in range(5)])
    definition = channel_definition(channel)
    counts = np.zeros(len(frame), dtype=int)
    for name in ("btag0", "btag1", "btag2", "btag3p"):
        counts += mask(frame, apply_region(definition.baseline, definition.region(name)).cuts)
    assert counts.tolist() == [1] * 5


def test_control_builder_data_routing_and_no_tau_estimator():
    analysis = build(config())
    analysis.validate()
    assert analysis.estimator is None and analysis.ml is None and not analysis.categories
    for channel in ("em", "mm", "ee"):
        prefix = "EGamma_" if channel == "ee" else "SingleMuon_"
        assert len(analysis.samples_for("data", channel)) == 4
        assert all(s.nick.startswith(prefix) for s in analysis.samples_for("data", channel))
        columns = selection_columns(analysis.channel(channel).baseline)
        for process in analysis.processes:
            columns |= selection_columns(process.selection_for(channel))
        assert not any("tau" in column or "gen_match" in column for column in columns)
    assert analysis.process("TT").selection.weights == {"top_pt": "topPtReweightWeight"}
    assert "yield" in analysis.control_variables


@pytest.mark.parametrize("channel,expected", [("em", 2*3*5*7), ("mm", 2*3*5*7), ("ee", 2*3*5*7)])
def test_both_lepton_legs_weighted(channel, expected):
    branches = {
        "em": {"id_wgt_ele_1": 2, "reco_wgt_ele_1": 3, "id_wgt_mu_2": 5, "iso_wgt_mu_2": 7},
        "mm": {"id_wgt_mu_1": 2, "iso_wgt_mu_1": 3, "id_wgt_mu_2": 5, "iso_wgt_mu_2": 7},
        "ee": {"id_wgt_ele_1": 2, "reco_wgt_ele_1": 3, "id_wgt_ele_2": 5, "reco_wgt_ele_2": 7},
    }[channel]
    frame = pd.DataFrame([branches | {"puweight": 1, "btag_weight_upart": 1,
                          "trg_wgt_single_mu24": 1, "trg_wgt_single_ele32": 1}])
    assert weight(frame, channel_definition(channel).baseline.weights).tolist() == [expected]


@pytest.mark.parametrize("channel", ["et", "mt", "tt"])
def test_expanded_tau_skim_contains_every_control_region(channel):
    definition = tau_channel(channel, control_regions=True)
    assert definition.baseline == tau_channel(channel).baseline
    # Exercise charges, b-tag bins, tau pass/fail, high mT and lepton isolation.
    rows = []
    for charge in (-1, 1):
        for btags in (0, 1, 2, 3):
            for passed in (0, 1):
                for mt in (20, 80, 90):
                    for iso in (.1, .15, .3, .5):
                        row = {column: 1. for column in branches(channel)}
                        row.update(q_1=1, q_2=charge, n_bjets=btags, n_jets=max(2, btags),
                                   bpair_pt_2=30 if btags else -999, pt_1=50, pt_2=45, mt_1=mt, iso_1=iso,
                                   extraelec_veto=0, extramuon_veto=0, dilepton_veto=0,
                                   id_tau_vsJet_Medium_2=passed)
                        rows.append(row)
    frame = pd.DataFrame(rows)
    skim = mask(frame, definition.skim.cuts)
    for region in definition.regions:
        accepted = mask(frame, apply_region(definition.baseline, region).cuts)
        assert not np.any(accepted & ~skim), region.name
    assert "btag0_os_pass" in {r.name for r in definition.regions}
    if channel != "tt":
        assert mask(frame, apply_region(definition.baseline, definition.region("w_highmt_pass")).cuts).any()
        assert mask(frame, apply_region(definition.baseline, definition.region("lepton_antiiso")).cuts).any()
    build_tau(config(channels=(channel,), control_regions=True)).validate()


@pytest.mark.parametrize("switches", [{"jet_fakes": "ff"}, {"nn_friend": True}, {"production": "old"}])
def test_tau_only_or_unknown_switches_rejected_for_dileptons(switches):
    with pytest.raises(AnalysisError):
        build(config(**switches))


@pytest.mark.parametrize("channel", ["em", "mm", "ee"])
def test_control_columns_match_generated_crown_output(channel):
    assert required_columns(build(config()), channel) <= branches(channel)


def test_v4_run_configs_share_ntuples_and_skims_but_not_outputs():
    from shapesmith.config import load_analysis, load_config
    tau = load_config(REPO / "configs" / "sm2018_binned_v4.yaml")
    dilepton = load_config(REPO / "configs" / "sm2018_binned_v4_dilepton.yaml")
    assert tau.ntuples == dilepton.ntuples and tau.skim_dir == dilepton.skim_dir
    assert tau.output_dir != dilepton.output_dir and not set(tau.channels) & set(dilepton.channels)
    assert tau.switches["control_regions"] is True
    for cfg in (tau, dilepton):
        load_analysis(cfg.model_copy(update={"sample_database": DATABASE})).validate()


def test_tau_raw_controls_require_mc_fakes():
    with pytest.raises(AnalysisError, match="control_regions.*jet_fakes"):
        build_tau(config(channels=("mt",), control_regions=True, jet_fakes="ff"))


def test_zero_b_control_accepts_missing_bb_pair_and_keeps_pass_fail_disjoint():
    definition = tau_channel("mt", control_regions=True)
    row = {column: 1. for column in branches("mt")}
    row.update(q_1=1, q_2=-1, n_bjets=0, n_jets=2, bpair_pt_2=-999,
               pt_1=50, pt_2=45, mt_1=90, iso_1=.1,
               extraelec_veto=0, extramuon_veto=0, dilepton_veto=0)
    frame = pd.DataFrame([row, row | {"id_tau_vsJet_Medium_2": 0}, row | {"mt_1": 80}])
    assert mask(frame, definition.skim.cuts).all()
    assert mask(frame, apply_region(definition.baseline, definition.region("w_highmt_pass")).cuts).tolist() == [True, False, False]
    assert mask(frame, apply_region(definition.baseline, definition.region("w_highmt_fail")).cuts).tolist() == [False, True, False]


def test_fail_region_weights_do_not_apply_passing_tau_or_muon_iso_sf():
    definition = tau_channel("mt", control_regions=True)
    fail = apply_region(definition.baseline, definition.region("btag1_os_fail"))
    frame = pd.DataFrame([{"gen_match_2": 5, "id_tau_vsJet_Medium_2": 0, "id_wgt_tau_vsJet_Medium_2": .8}])
    assert weight(frame, {"tau_id": fail.weights["tau_id"]}).tolist() == [1.]
    antiiso = apply_region(definition.baseline, definition.region("lepton_antiiso"))
    assert weight(pd.DataFrame({"event": [1]}), {"iso": antiiso.weights["iso"]}).tolist() == [1.]


def test_tau_data_and_mc_control_histograms_from_parquet(tmp_path):
    from dataclasses import replace
    from shapesmith.histograms import HistKey, run_hist
    from shapesmith.io.skims import write_skim
    from shapesmith.skim import columns_for_sample

    cfg = config(channels=("mt",), control_regions=True).model_copy(
        update={"skim_dir": tmp_path / "skim", "output_dir": tmp_path / "out", "workers": 1})
    analysis = build_tau(cfg)
    data_sample = analysis.samples_for("data", "mt")[0]
    tt_sample = analysis.samples_for("TT", "mt")[0]
    analysis = replace(analysis, samples=(data_sample, tt_sample),
                       processes=(analysis.process("data"), analysis.process("TTT")), lumi_pb=1.)
    for sample in analysis.samples:
        columns = columns_for_sample(analysis, "mt", sample)
        if sample.kind == "data":
            assert "gen_match_2" not in columns and "id_wgt_tau_vsJet_Medium_2" not in columns
        row = {column: 1. for column in columns}
        row.update(q_1=1, q_2=-1, n_bjets=0, n_jets=2, bpair_pt_2=-999,
                   pt_1=50, pt_2=45, mt_1=90, iso_1=.1,
                   extraelec_veto=0, extramuon_veto=0, dilepton_veto=0,
                   sample_nick=sample.nick, norm_weight=1., is_data=sample.kind == "data",
                   is_mc=sample.kind == "mc", is_embedding=False)
        if sample.kind == "mc":
            row.update(gen_match_1=4, gen_match_2=5, id_wgt_tau_vsJet_Medium_2=.8)
        frame = pd.DataFrame([row, row | {"id_tau_vsJet_Medium_2": 0}])
        write_skim(frame, cfg.skim_dir / "mt" / sample.nick / "events.parquet")
    hists = run_hist(cfg, analysis, ["mt"], control=True, variables=["mt_1"], systematics=False,
                     processes=None, output=cfg.output_dir / "controls.root",
                     regions=["w_highmt_pass", "w_highmt_fail"])
    for process, state, expected in (("data", "pass", 1), ("data", "fail", 1),
                                     ("TTT", "pass", .8), ("TTT", "fail", 1)):
        h = hists.get(HistKey("mt", "inclusive", process, f"w_highmt_{state}", "Nominal", "mt_1"))
        assert h.sum() == pytest.approx(expected)


@pytest.mark.parametrize("channel", ["em", "mm", "ee"])
def test_dilepton_control_histograms_and_plot_from_parquet(channel, tmp_path):
    from dataclasses import replace
    from shapesmith.histograms import HistKey, run_hist
    from shapesmith.io.skims import write_skim
    from shapesmith.plotting.stack import run_plot
    from shapesmith.skim import columns_for_sample

    cfg = config(channels=(channel,)).model_copy(
        update={"skim_dir": tmp_path / "skim", "output_dir": tmp_path / "out", "workers": 1})
    analysis = build(cfg)
    data_sample = analysis.samples_for("data", channel)[0]
    tt_sample = analysis.samples_for("TT", channel)[0]
    analysis = replace(analysis, samples=(data_sample, tt_sample),
                       processes=(analysis.process("data"), analysis.process("TT")), lumi_pb=1.)
    for sample in analysis.samples:
        row = {column: 1. for column in columns_for_sample(analysis, channel, sample)}
        row.update(event(), sample_nick=sample.nick, norm_weight=1. if sample.kind == "data" else 2.,
                   is_data=sample.kind == "data", is_mc=sample.kind == "mc", is_embedding=False)
        frame = pd.DataFrame([row | {"n_bjets": 0}, row | {"n_bjets": 2},
                              row | {"n_bjets": 2, "q_2": 1}])
        write_skim(frame, cfg.skim_dir / channel / sample.nick / "events.parquet")
    hists = run_hist(cfg, analysis, [channel], control=True, variables=["yield"], systematics=False,
                     processes=None, output=cfg.output_dir / "controls.root", regions=["btag2"])
    assert hists.get(HistKey(channel, "inclusive", "data", "btag2", "Nominal", "yield")).sum() == 1.
    assert hists.get(HistKey(channel, "inclusive", "TT", "btag2", "Nominal", "yield")).sum() == 2.
    paths = run_plot(hists, analysis, [channel], control=True, category=None, variables=["yield"],
                     output_dir=cfg.output_dir / "plots", region="btag2")
    assert {path.suffix for path in paths} == {".pdf", ".png"}
    assert all(path.parent.name == "btag2" and path.stat().st_size > 0 for path in paths)
