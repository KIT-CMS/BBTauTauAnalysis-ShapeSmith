"""Tau-channel selections and weights: every column exists, and the regions act on the processes as intended."""
import numpy as np
import pandas as pd
import pytest
from shapesmith.events import select
from shapesmith.expressions import columns_in, columns_of, mask, product

from bbtautau_shapesmith import cuts, weights
from bbtautau_shapesmith.constants import DY_NJETS_BTAG, FF_COLUMNS, FF_RAW_COLUMNS, LT_CHANNELS, TAU_CHANNELS
from tests.helpers import branches, embedding_branches
from tests.test_analysis import analysis_for


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_selection_and_weight_columns_exist(channel):
    assert columns_of(cuts.baseline_cuts(channel).values()) <= branches(channel)
    assert columns_of(cuts.skim_cuts(channel, control_regions=True).values()) <= branches(channel)
    assert columns_of(cuts.genmatch_cuts(channel).values()) <= branches(channel)
    assert columns_of(weights.mc_weights(channel).values()) <= branches(channel)
    for region in cuts.regions(channel, "ff") + cuts.regions(channel, "ff", "raw") + cuts.regions(channel, "mc") + cuts.diagnostic_regions(channel):
        exprs = [*region.replace_cuts.values(), *region.add_weights.values(), *region.replace_weights.values()]
        assert columns_of(exprs) <= branches(channel) | set(FF_COLUMNS.values()) | set(FF_RAW_COLUMNS.values()), region.name


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_raw_fake_factor_weight_treats_the_legs_like_the_corrected_one(channel):
    corrected, raw = weights.fake_factor_weight(channel), weights.fake_factor_weight(channel, "raw")
    legs = ("lt",) if channel in LT_CHANNELS else ("tt_1", "tt_2")
    assert columns_in(raw) - columns_in(corrected) == {FF_RAW_COLUMNS[leg] for leg in legs}
    assert columns_in(corrected) - columns_in(raw) == {FF_COLUMNS[leg] for leg in legs}
    # the same values in both column sets give the same weight: 0.5 per failing leg in tt
    rows = [{"id_tau_vsJet_Medium_1": pass_1, "id_tau_vsJet_Medium_2": pass_2} for pass_1 in (0, 1) for pass_2 in (0, 1)]
    frame = pd.DataFrame(rows)
    for leg, value in zip(legs, (.3, .7)):
        frame[FF_COLUMNS[leg]] = frame[FF_RAW_COLUMNS[leg]] = value
    expected = [.3] * 4 if channel != "tt" else [.5, .15, .35, 0.]
    assert product(frame, [raw]).tolist() == pytest.approx(expected)
    assert product(frame, [corrected]).tolist() == pytest.approx(expected)


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_embedding_weights_follow_the_part_a_contract(channel):
    emb = weights.embedding_weights(channel)
    assert columns_of(emb.values()) <= embedding_branches(channel)
    assert not set(emb) & set(weights.mc_weights(channel))  # own names: MC weight replacements never reach EMB
    assert ("iso_wgt_ele_1" in emb.get("emb_iso", "")) == (channel == "et")  # the electron iso SF exists in embedding only
    assert {"emb_vs_mu", "emb_vs_ele", "emb_tau_id", "emb_trigger"} <= set(emb)


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_tau_eta_in_et_and_mt(channel):
    expected = "(abs(eta_2) < 2.3)" if channel in LT_CHANNELS else None
    for selection in (cuts.baseline_cuts(channel), cuts.skim_cuts(channel), cuts.skim_cuts(channel, control_regions=True)):
        assert selection.get("tau_eta") == expected
    frame = pd.DataFrame({"eta_2": [-2.31, -2.29, 2.29, 2.3]})
    assert mask(frame, [cuts.baseline_cuts("mt")["tau_eta"]]).tolist() == [False, True, True, False]


def test_dy_correction_weight_takes_the_table_value_of_each_cell():
    frame = pd.DataFrame([{"n_bjets": b, "n_jets": j} for b in range(4) for j in (2, 3, 4, 6)] + [{"n_bjets": 1, "n_jets": 1}])
    table = DY_NJETS_BTAG
    expected = [table[min(b, 2)][min(j, 4) - 2] for b in range(4) for j in (2, 3, 4, 6)] + [table[1][0]]
    assert product(frame, [weights.dy_njets_btag_weight()]).tolist() == pytest.approx(expected)
    assert columns_in(weights.dy_njets_btag_weight()) == {"n_bjets", "n_jets"} <= branches("tt") & branches("mm")


def test_light_lepton_scale_factors():
    """et: electron ID (wp90iso, covering the isolation) and reconstruction SF in MC; the embedding keeps its own
    ID and iso SFs and has no reconstruction SF."""
    assert weights.lepton_weights("et") == {"id": "id_wgt_ele_1", "reco": "reco_wgt_ele_1"}
    assert weights.lepton_weights("mt") == {"id": "id_wgt_mu_1", "iso": "iso_wgt_mu_1"} and weights.lepton_weights("tt") == {}
    assert "reco_wgt_ele_1" not in columns_of(weights.embedding_weights("et").values())


def test_regions_replace_mc_weights_only():
    channel = analysis_for("mc", embedding=True, control_regions=True).channel("mt")
    for region_name in ("btag1_os_fail", "lepton_antiiso"):
        region = channel.region(region_name)
        for name in ("TTL", "EMB", "data"):
            process = channel.process(name)
            nominal = select(channel, process, channel.region("nominal"))[1]
            varied = select(channel, process, region)[1]
            if name == "TTL":
                assert varied == [region.replace_weights.get(k, v) for k, v in process.selection.weights.items()] and varied != nominal
            else:
                assert varied == nominal, (region_name, name)


def cut_columns(definition, channel: str) -> dict[str, float]:
    """1.0 for every column the skim, the baseline and the regions of the channel cut on (all of them branches)."""
    exprs = [*definition.skim.values(), *definition.cuts.values(), *(expr for r in definition.regions for expr in r.replace_cuts.values())]
    columns = columns_of(exprs)
    assert columns <= branches(channel)
    return dict.fromkeys(columns, 1.0)


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_expanded_tau_skim_contains_every_control_region(channel):
    definition = analysis_for("mc", control_regions=True).channel(channel)
    columns = cut_columns(definition, channel)
    assert definition.cuts == cuts.baseline_cuts(channel)
    # Exercise charges, b-tag bins, tau pass/fail, high mT and lepton isolation.
    rows = []
    for charge in (-1, 1):
        for btags in (0, 1, 2, 3):
            for passed in (0, 1):
                for mt in (20, 80, 90):
                    for iso in (.1, .15, .3, .5):
                        row = dict(columns)
                        row.update(q_1=1, q_2=charge, n_bjets=btags, n_jets=max(2, btags),
                                   bpair_pt_2=30 if btags else -999, pt_1=50, pt_2=45, mt_1=mt, iso_1=iso,
                                   extraelec_veto=0, extramuon_veto=0, dilepton_veto=0,
                                   id_tau_vsJet_Medium_2=passed)
                        rows.append(row)
    frame = pd.DataFrame(rows)
    skim = mask(frame, definition.skim.values())
    for region in definition.regions:
        accepted = mask(frame, {**definition.cuts, **region.replace_cuts}.values())
        assert not np.any(accepted & ~skim), region.name
    names = {r.name for r in definition.regions}
    assert "btag0_os_pass" in names and ("w_highmt_pass" in names) == (channel != "tt")


def test_zero_b_control_accepts_missing_bb_pair_and_keeps_pass_fail_disjoint():
    definition = analysis_for("mc", control_regions=True).channel("mt")
    row = cut_columns(definition, "mt")
    row.update(q_1=1, q_2=-1, n_bjets=0, n_jets=2, bpair_pt_2=-999, pt_1=50, pt_2=45, mt_1=90, iso_1=.1,
               extraelec_veto=0, extramuon_veto=0, dilepton_veto=0)
    frame = pd.DataFrame([row, row | {"id_tau_vsJet_Medium_2": 0}, row | {"mt_1": 80}])
    assert mask(frame, definition.skim.values()).all()
    for state, expected in (("pass", [True, False, False]), ("fail", [False, True, False])):
        region = definition.region(f"w_highmt_{state}")
        assert mask(frame, {**definition.cuts, **region.replace_cuts}.values()).tolist() == expected


def test_fail_region_weights_do_not_apply_passing_tau_or_muon_iso_sf():
    frame = pd.DataFrame([{"gen_match_2": 5, "id_tau_vsJet_Medium_2": 0, "id_wgt_tau_vsJet_Medium_2": .8}])
    assert product(frame, weights.fail_tau_weights("mt").values()).tolist() == [1.]
    assert columns_in(cuts.diagnostic_regions("mt")[-1].replace_weights["iso"]) == set()
