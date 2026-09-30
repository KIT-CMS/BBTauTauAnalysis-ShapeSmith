"""Tau-channel selections and weights: every column exists, and the regions act on the processes as intended."""
import numpy as np
import pandas as pd
import pytest
from shapesmith.events import select
from shapesmith.expressions import columns_in, columns_of, mask, product

from bbtautau_shapesmith import cuts, weights
from bbtautau_shapesmith.constants import FF_COLUMNS, TAU_CHANNELS
from tests.helpers import branches, embedding_branches
from tests.test_analysis import analysis_for


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_selection_and_weight_columns_exist(channel):
    assert columns_of(cuts.baseline_cuts(channel).values()) <= branches(channel)
    assert columns_of(cuts.skim_cuts(channel, control_regions=True).values()) <= branches(channel)
    assert columns_of(cuts.genmatch_cuts(channel).values()) <= branches(channel)
    assert columns_of(weights.mc_weights(channel).values()) <= branches(channel)
    for region in cuts.regions(channel, "ff") + cuts.regions(channel, "mc") + cuts.diagnostic_regions(channel):
        exprs = [*region.replace_cuts.values(), *region.add_weights.values(), *region.replace_weights.values()]
        assert columns_of(exprs) <= branches(channel) | set(FF_COLUMNS.values()), region.name


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_embedding_weights_follow_the_part_a_contract(channel):
    emb = weights.embedding_weights(channel)
    assert columns_of(emb.values()) <= embedding_branches(channel)
    assert not set(emb) & set(weights.mc_weights(channel))  # own names: MC weight replacements never reach EMB
    assert ("iso_wgt_ele_1" in emb.get("emb_iso", "")) == (channel == "et")  # the electron iso SF exists in embedding only
    assert {"emb_vs_mu", "emb_vs_ele", "emb_tau_id", "emb_trigger"} <= set(emb)


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


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_expanded_tau_skim_contains_every_control_region(channel):
    definition = analysis_for("mc", control_regions=True).channel(channel)
    assert definition.cuts == cuts.baseline_cuts(channel)
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
    skim = mask(frame, definition.skim.values())
    for region in definition.regions:
        accepted = mask(frame, {**definition.cuts, **region.replace_cuts}.values())
        assert not np.any(accepted & ~skim), region.name
    names = {r.name for r in definition.regions}
    assert "btag0_os_pass" in names and ("w_highmt_pass" in names) == (channel != "tt")


def test_zero_b_control_accepts_missing_bb_pair_and_keeps_pass_fail_disjoint():
    definition = analysis_for("mc", control_regions=True).channel("mt")
    row = {column: 1. for column in branches("mt")}
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
