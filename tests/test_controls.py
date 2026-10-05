"""Control channels and variables tested on events, including histograms filled from Parquet skims."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
from shapesmith.config import load_analysis, load_config
from shapesmith.expressions import columns_in, columns_of, mask, product
from shapesmith.fill import run_hist
from shapesmith.histogram import HistKey
from shapesmith.model import AnalysisError
from shapesmith.plotting.stack import run_plot
from shapesmith.skim import needed_columns
from shapesmith.store import write_skim

from bbtautau_shapesmith.constants import DILEPTON_CHANNELS, NN_COLUMNS, TAU_CHANNELS
from bbtautau_shapesmith.dilepton import build
from bbtautau_shapesmith.variables import control_variables
from tests.helpers import DATABASE, REPO, branches, config
from tests.test_analysis import analysis_for

DILEPTON = "bbtautau_shapesmith.dilepton:build"


def dilepton_channel(channel):
    return build(config([channel], DILEPTON)).channel(channel)


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
    assert mask(frame, dilepton_channel(channel).cuts.values()).tolist() == expected


@pytest.mark.parametrize("channel", DILEPTON_CHANNELS)
def test_btag_regions_partition_two_jet_events(channel):
    frame = pd.DataFrame([event(n_bjets=n, n_jets=max(2, n)) for n in range(5)])
    definition = dilepton_channel(channel)
    counts = np.zeros(len(frame), dtype=int)
    for name in ("btag0", "btag1", "btag2", "btag3p"):
        counts += mask(frame, {**definition.cuts, **definition.region(name).replace_cuts}.values())
    assert counts.tolist() == [1] * 5


def test_dilepton_data_routing_unsplit_mc_and_no_tau_estimate():
    analysis = build(config(DILEPTON_CHANNELS, DILEPTON))
    for name in DILEPTON_CHANNELS:
        channel = analysis.channel(name)
        assert channel.estimators == () and channel.categories == ()
        prefix = "EGamma_" if name == "ee" else "SingleMuon_"
        data = channel.samples_of("data")
        assert data and all(s.nick.startswith(prefix) for s in data)
        assert {"DY", "TT", "ST", "VV", "TTV", "W", "EWK", "ggH125", "qqH125", "ttH125", "VH125"} <= {p.name for p in channel.processes}
        columns = columns_of(channel.cuts.values()) | columns_of(e for p in channel.processes for e in (*p.selection.cuts.values(), *p.selection.weights.values()))
        assert not any("tau" in column or "gen_match" in column for column in columns)
        assert "yield" in channel.variables
    assert analysis.ml is None


@pytest.mark.parametrize("channel,expected", [("em", 2*3*5*7), ("mm", 2*3*5*7), ("ee", 2*3*5*7)])
def test_both_lepton_legs_weighted(channel, expected):
    columns = {
        "em": {"id_wgt_ele_1": 2, "reco_wgt_ele_1": 3, "id_wgt_mu_2": 5, "iso_wgt_mu_2": 7},
        "mm": {"id_wgt_mu_1": 2, "iso_wgt_mu_1": 3, "id_wgt_mu_2": 5, "iso_wgt_mu_2": 7},
        "ee": {"id_wgt_ele_1": 2, "reco_wgt_ele_1": 3, "id_wgt_ele_2": 5, "reco_wgt_ele_2": 7},
    }[channel]
    frame = pd.DataFrame([columns | {"puweight": 1, "btag_weight_upart": 1, "trg_wgt_single_mu24": 1, "trg_wgt_single_ele32": 1}])
    assert product(frame, dilepton_channel(channel).process("DY").selection.weights.values()).tolist() == [expected]


@pytest.mark.parametrize("switches", [{"jet_fakes": "ff"}, {"nn_friend": True}, {"production": "old"}])
def test_tau_only_or_unknown_switches_rejected_for_dileptons(switches):
    with pytest.raises(AnalysisError):
        build(config(DILEPTON_CHANNELS, DILEPTON, **switches))


@pytest.mark.parametrize("channel", DILEPTON_CHANNELS)
def test_control_columns_match_generated_crown_output(channel):
    definition = dilepton_channel(channel)
    for sample in definition.samples:
        assert needed_columns(definition, sample) <= branches(channel)


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_control_variables_exist(channel):
    for name, variable in control_variables(nn_friend=True).items():
        assert variable.name == name
        assert columns_in(variable.expr) <= branches(channel) | NN_COLUMNS, (name, columns_in(variable.expr) - branches(channel))
        assert len(variable.edges) >= 2 and all(b > a for a, b in zip(variable.edges, variable.edges[1:]))
    assert {"m_vis", "n_bjets", "bpair_m_inv", "mass_tautaubb", "sum_deltaR_tt_bb", "NN_score"} <= set(control_variables(nn_friend=True))
    assert "NN_score" not in control_variables()


def test_v4_run_configs_share_ntuples_and_skims_but_not_outputs():
    tau = load_config(REPO / "configs" / "sm2018_binned_v4.yaml")
    dilepton = load_config(REPO / "configs" / "sm2018_binned_v4_dilepton.yaml")
    assert tau.ntuples == dilepton.ntuples and tau.skim_dir == dilepton.skim_dir
    assert tau.output_dir != dilepton.output_dir and not set(tau.channels) & set(dilepton.channels)
    assert tau.switches["control_regions"] is True
    for cfg in (tau, dilepton):
        load_analysis(cfg.model_copy(update={"sample_database": DATABASE}))


def test_raw_ff_run_config_is_the_ff_run_config_with_raw_fake_factors():
    corrected = load_config(REPO / "configs" / "sm2018_binned_v5_ff.yaml")
    raw = load_config(REPO / "configs" / "sm2018_binned_v5_rawff.yaml")
    assert raw.switches == corrected.switches | {"ff_type": "raw", "shape_systematics": False}
    assert (raw.ntuples, raw.channels, raw.sample_database) == (corrected.ntuples, corrected.channels, corrected.sample_database)
    assert all(getattr(raw, name) != getattr(corrected, name) for name in ("skim_dir", "output_dir", "ml_dir"))
    load_analysis(raw.model_copy(update={"sample_database": DATABASE}))


def only(analysis, channel_name, *processes):
    """The analysis reduced to one data and one MC sample and the given processes, with lumi 1."""
    channel = analysis.channel(channel_name)
    chosen = [channel.process(name) for name in processes]
    kept = tuple(channel.samples_of(p.group)[0] for p in chosen)
    channel = dataclasses.replace(channel, samples=kept, processes=tuple(chosen), estimators=())
    return dataclasses.replace(analysis, lumi_pb=1., channels={channel_name: channel})


def write_rows(cfg, channel, rows_of):
    for sample in channel.samples:
        row = {column: 1. for column in needed_columns(channel, sample)}
        row.update(sample_nick=sample.nick, norm_weight=1. if sample.kind == "data" else 2., is_data=sample.kind == "data",
                   is_mc=sample.kind == "mc", is_embedding=False)
        write_skim(pd.DataFrame(rows_of(sample, row)), cfg.skim_dir / channel.name / sample.nick / "events.parquet")


def test_tau_data_and_mc_control_histograms_from_parquet(tmp_path):
    cfg = config(["mt"], control_regions=True).model_copy(update={"skim_dir": tmp_path / "skim", "output_dir": tmp_path / "out", "workers": 1})
    analysis = only(analysis_for("mc", control_regions=True), "mt", "data", "TTT")
    channel = analysis.channel("mt")
    assert "gen_match_2" not in needed_columns(channel, channel.samples_of("data")[0])

    def rows(sample, row):
        row.update(q_1=1, q_2=-1, n_bjets=0, n_jets=2, bpair_pt_2=-999, pt_1=50, pt_2=45, mt_1=90, iso_1=.1,
                   extraelec_veto=0, extramuon_veto=0, dilepton_veto=0, norm_weight=1.)
        if sample.kind == "mc":
            row.update(gen_match_1=4, gen_match_2=5, id_wgt_tau_vsJet_Medium_2=.8)
        return [row, row | {"id_tau_vsJet_Medium_2": 0}]

    write_rows(cfg, channel, rows)
    hists = run_hist(cfg, analysis, ["mt"], control=True, variables=["mt_1"], systematics=False, processes=None,
                     output=cfg.output_dir / "controls.root", regions=["w_highmt_pass", "w_highmt_fail"])
    for process, state, expected in (("data", "pass", 1), ("data", "fail", 1), ("TTT", "pass", .8), ("TTT", "fail", 1)):
        assert hists[HistKey("mt", "inclusive", process, f"w_highmt_{state}", "Nominal", "mt_1")].sum() == pytest.approx(expected)


@pytest.mark.parametrize("channel_name", DILEPTON_CHANNELS)
def test_dilepton_control_histograms_and_plot_from_parquet(channel_name, tmp_path):
    cfg = config([channel_name], DILEPTON).model_copy(update={"skim_dir": tmp_path / "skim", "output_dir": tmp_path / "out", "workers": 1})
    analysis = only(build(cfg), channel_name, "data", "TT")
    channel = analysis.channel(channel_name)
    write_rows(cfg, channel, lambda sample, row: [row | event() | {"n_bjets": 0}, row | event() | {"n_bjets": 2}, row | event() | {"n_bjets": 2, "q_2": 1}])
    hists = run_hist(cfg, analysis, [channel_name], control=True, variables=["yield"], systematics=False, processes=None,
                     output=cfg.output_dir / "controls.root", regions=["btag2"])
    assert hists[HistKey(channel_name, "inclusive", "data", "btag2", "Nominal", "yield")].sum() == 1.
    assert hists[HistKey(channel_name, "inclusive", "TT", "btag2", "Nominal", "yield")].sum() == 2.
    paths = run_plot(hists, analysis, [channel_name], control=True, category=None, variables=["yield"],
                     output_dir=cfg.output_dir / "plots", region="btag2")
    assert {path.suffix for path in paths} == {".pdf", ".png"}
    assert all(path.parent.name == "btag2" and path.stat().st_size > 0 for path in paths)
