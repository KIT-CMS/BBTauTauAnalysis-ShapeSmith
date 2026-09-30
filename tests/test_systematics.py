import pytest
from shapesmith.events import select
from shapesmith.histogram import part_of
from shapesmith.model import ColumnVariation, VariationSum, WeightVariation

from bbtautau_shapesmith.constants import FF_SHIFTS_LT, FF_SHIFTS_TT
from bbtautau_shapesmith.systematics import embedding_variation_sums, embedding_variations, ff_variations
from bbtautau_shapesmith.weights import fake_factor_weight
from tests.test_analysis import COMBINATIONS, analysis_for


def column_variations(channel):
    return [v for v in channel.variations if isinstance(v, ColumnVariation)]


@pytest.mark.parametrize("jet_fakes,embedding", COMBINATIONS)
def test_column_variations_come_with_their_switch(jet_fakes, embedding):
    analysis = analysis_for(jet_fakes, embedding)
    for name, channel in analysis.channels.items():
        expected = (embedding_variations(name) if embedding else ()) + (ff_variations(name) if jet_fakes == "ff" else ())
        assert tuple(column_variations(channel)) == expected
        assert sum(isinstance(v, WeightVariation) for v in channel.variations) == 80  # the b-tag components


@pytest.mark.parametrize("jet_fakes,embedding", COMBINATIONS)
def test_shape_systematics_off_declares_no_column_variation(jet_fakes, embedding):
    analysis = analysis_for(jet_fakes, embedding, shape_systematics=False)
    assert not any(column_variations(channel) for channel in analysis.channels.values())


@pytest.mark.parametrize("channel,count", [("et", 24), ("mt", 24), ("tt", 16)])
def test_embedding_variations(channel, count):
    variations = embedding_variations(channel)
    assert len(variations) == count and all(v.applies_to == ("embedding",) and v.regions is None for v in variations)
    wp = "vsEleTight" if channel == "et" else "vsEleVVLoose"
    names = {v.name: v.suffix for v in variations}
    assert names[f"CMS_scale_t_emb_dm0_{wp}_2018Down"] == "__CMS_scale_t_emb_DeepTau2018v2p5_DM0_2018Down"
    assert names[f"CMS_eff_t_emb_dm1_pt40toInf_{wp}_2018Up"] == "__CMS_eff_t_emb_DeepTau2018v2p5_VSjet_DM1_pt40toInf_2018Up"
    # DM10 and DM11 are the parts of one nuisance
    assert f"CMS_scale_t_emb_dm1011_{wp}_2018Up" not in names
    assert names[part_of(f"CMS_scale_t_emb_dm1011_{wp}_2018Up", "dm10")] == "__CMS_scale_t_emb_DeepTau2018v2p5_DM10_2018Up"
    assert names[part_of(f"CMS_scale_t_emb_dm1011_{wp}_2018Down", "dm11")] == "__CMS_scale_t_emb_DeepTau2018v2p5_DM11_2018Down"
    part = part_of(f"CMS_eff_t_emb_dm1011_pt20to40_{wp}_2018Down", "dm11")
    assert (part in names) == (channel != "tt")  # tt: both taus above 40 GeV


@pytest.mark.parametrize("channel", ["et", "mt", "tt"])
def test_embedding_variation_sums_join_dm10_and_dm11(channel):
    wp = "vsEleTight" if channel == "et" else "vsEleVVLoose"
    pt_bins = ("40toInf",) if channel == "tt" else ("20to40", "40toInf")
    expected = (VariationSum(f"CMS_scale_t_emb_dm1011_{wp}_2018", ("dm10", "dm11")),) + tuple(
        VariationSum(f"CMS_eff_t_emb_dm1011_pt{pt}_{wp}_2018", ("dm10", "dm11")) for pt in pt_bins
    )
    assert embedding_variation_sums(channel) == expected


@pytest.mark.parametrize("channel,keys", [("et", FF_SHIFTS_LT), ("mt", FF_SHIFTS_LT), ("tt", FF_SHIFTS_TT)])
def test_ff_variations(channel, keys):
    variations = ff_variations(channel)
    assert len(variations) == 2 * len(keys) and all(v.applies_to == ("data", "mc", "embedding") for v in variations)
    names = {v.name: v.suffix for v in variations}
    assert names[f"CMS_ff_process_fractionsfrac_QCD_{channel}_2018Up"] == "__process_fractionsfrac_QCDUp"
    assert names[f"CMS_ff_QCD_non_closure_CorrSystMCShift_{channel}_2018Down"] == "__QCD_non_closure_CorrSystMCShiftDown"


def test_ff_shift_keys():
    assert len(FF_SHIFTS_LT) == len(set(FF_SHIFTS_LT)) == 17
    subleading = FF_SHIFTS_TT[len(FF_SHIFTS_LT):]
    assert FF_SHIFTS_TT[:len(FF_SHIFTS_LT)] == FF_SHIFTS_LT and len(set(subleading)) == 17
    # "_subleading" follows the process or correction name of the leading key
    assert sorted(key.replace("_subleading", "", 1) for key in subleading) == sorted(FF_SHIFTS_LT)


def test_tt_ff_shift_moves_the_fake_factor_of_its_leg():
    channel = analysis_for("ff").channel("tt")
    variation = next(v for v in channel.variations if v.name == "CMS_ff_QCD_subleadingStatShift_tt_2018Up")
    available = {"fake_factor_1", "fake_factor_2", "fake_factor_1__QCDStatShiftUp", "fake_factor_2__QCD_subleadingStatShiftUp"}
    weights = select(channel, channel.process("data"), channel.region("anti_iso"), variation, available)[1]
    assert weights == [fake_factor_weight("tt").replace("fake_factor_2", "fake_factor_2__QCD_subleadingStatShiftUp")]
