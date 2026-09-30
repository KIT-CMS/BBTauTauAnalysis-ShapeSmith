import pytest
from shapesmith.events import select
from shapesmith.model import ColumnVariation, WeightVariation

from bbtautau_shapesmith.constants import FF_SHIFTS_LT, FF_SHIFTS_TT
from bbtautau_shapesmith.systematics import embedding_variations, ff_variations
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


@pytest.mark.parametrize("channel,count", [("et", 18), ("mt", 18), ("tt", 12)])
def test_embedding_variations(channel, count):
    variations = embedding_variations(channel)
    assert len(variations) == count and all(v.applies_to == ("embedding",) and v.regions is None for v in variations)
    wp = "vsEleTight" if channel == "et" else "vsEleVVLoose"
    names = {v.name: v.suffix for v in variations}
    assert names[f"CMS_scale_t_emb_dm1011_{wp}_2018Up"] == "__embTauEs3prongUp"
    assert names[f"CMS_scale_t_emb_dm0_{wp}_2018Down"] == "__embTauEs1prong0pizeroDown"
    assert names[f"CMS_eff_t_emb_dm1_pt40toInf_{wp}_2018Up"] == "__embVsJetTauDM1Pt40toInfUp"
    assert (f"CMS_eff_t_emb_dm1011_pt20to40_{wp}_2018Down" in names) == (channel != "tt")  # tt: both taus above 40 GeV


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
