"""The fake-factor measurement analysis: regions from the analysis selection, processes and subtractions, tables."""
import pytest
from shapesmith.model import AnalysisError
from shapesmith.skim import needed_columns
from shapesmith.validate import validate

from bbtautau_shapesmith import cuts
from bbtautau_shapesmith.constants import TAU_CHANNELS
from bbtautau_shapesmith.ff_measurement import build
from bbtautau_shapesmith.ff_tables import TABLES
from tests.helpers import EMBEDDING_LIST, branches, config, embedding_branches

T_PARTS, L_PARTS, J_PARTS = ("ZTT", "TTT", "STT", "VVT"), ("ZL", "TTL", "STL", "VVL"), ("ZJ", "TTJ", "STJ", "VVJ")


def measurement_analysis(embedding=False, channels=TAU_CHANNELS):
    sample_lists = ["sm2018_binned_v3", EMBEDDING_LIST] if embedding else ["sm2018_binned_v3"]
    analysis = build(config(channels, "bbtautau_shapesmith.ff_measurement:build", embedding=embedding, sample_lists=sample_lists))
    validate(analysis)
    return analysis


def region_cuts(channel, name: str) -> dict[str, str]:
    return {**channel.cuts, **channel.region(name).replace_cuts}


@pytest.mark.parametrize("embedding", [False, True])
def test_builds_for_every_tau_channel_and_reads_only_existing_columns(embedding):
    analysis = measurement_analysis(embedding)
    for name, channel in analysis.channels.items():
        for sample in channel.samples:
            available = embedding_branches(name) if sample.kind == "embedding" else branches(name) | {"npartons"}
            missing = needed_columns(channel, sample) - available
            assert not missing, (sample.nick, missing)
    assert analysis.signal is None and analysis.measurement.name == "fake_factors"


def test_the_tau_id_is_the_analysis_tau_id():
    analysis = measurement_analysis()
    mt, tt = analysis.channel("mt"), analysis.channel("tt")
    assert region_cuts(mt, "QCD_sr_like")["tau_iso"] == cuts.isolated(2) and region_cuts(mt, "QCD_ar_like")["tau_iso"] == cuts.anti_isolated(2)
    assert region_cuts(mt, "ttbar_sr")["tau_iso"] == cuts.baseline_cuts("mt")["tau_iso"]
    assert region_cuts(tt, "QCD_ar_like")["tau_iso"] == f"{cuts.anti_isolated(1)} & {cuts.isolated(2)}"
    assert region_cuts(tt, "QCD_subleading_ar_like")["tau_iso"] == f"{cuts.isolated(1)} & {cuts.anti_isolated(2)}"
    # the tt fractions: both taus anti-isolated for both legs (U6)
    both = f"{cuts.anti_isolated(1)} & {cuts.anti_isolated(2)}"
    assert region_cuts(tt, "process_fractions_ar")["tau_iso"] == region_cuts(tt, "process_fractions_subleading_ar")["tau_iso"] == both
    assert region_cuts(mt, "process_fractions_ar") == {**mt.cuts, "tau_iso": cuts.anti_isolated(2)}


def test_the_regions():
    analysis = measurement_analysis()
    mt, et, tt = (analysis.channel(c) for c in ("mt", "et", "tt"))
    qcd = region_cuts(mt, "QCD_sr_like")
    assert qcd["os"] == cuts.SAME_SIGN and qcd["mt_cut"] == "(mt_1 < 50)" and qcd["b_tagging"] == "(n_jets >= 2)"
    # the DR->SR regions: jvoss's lepton isolation below his n-tuples' 0.4 (U5), opposite sign for the correction
    assert region_cuts(mt, "QCD_orthogonal_sr_like")["lepton_iso"] == "(~((iso_1 >= 0.05) & (iso_1 <= 0.15))) & (iso_1 < 0.4)"
    assert "0.02" in region_cuts(et, "QCD_dr_sr_ar_like")["lepton_iso"] and region_cuts(et, "QCD_dr_sr_ar_like")["os"] == cuts.baseline_cuts("et")["os"]
    assert region_cuts(tt, "QCD_orthogonal_sr_like")["tau_iso"] == f"{cuts.isolated(1)} & {cuts.anti_isolated(2)}"
    # the ttbar scale regions invert the lepton vetoes as one cut
    scale = region_cuts(mt, "ttbar_scale_sr_like_ss")
    assert scale["lepton_vetoes"].startswith("~(") and "dilepton_veto" in scale["lepton_vetoes"] and scale["os"] == cuts.SAME_SIGN
    assert "dilepton_veto" not in region_cuts(tt, "ttbar_scale_ar_like")["lepton_vetoes"]
    assert not any("veto" in name for name in mt.skim)  # stored, not applied


@pytest.mark.parametrize("embedding", [False, True])
def test_the_subtractions(embedding):
    (leg,) = measurement_analysis(embedding, ("mt",)).measurement.legs["mt"]
    genuine = ("EMB",) if embedding else T_PARTS
    fakes = L_PARTS + J_PARTS + ("W",)
    assert leg.qcd.target == "data" and leg.qcd.subtract == genuine + fakes
    assert leg.qcd.dr_sr.subtract == T_PARTS + fakes  # TauFakeFactors DR_SR use_embedding: false
    assert leg.ttbar.target == "TTJ" and leg.ttbar.subtract == () and leg.ttbar.scale.subtract == genuine + fakes
    assert leg.fractions.subtract == genuine + fakes and leg.fractions.ttbar == "TTJ"


def test_the_tables():
    for channel, tables in TABLES.items():
        for name, table in tables.items():
            if name.startswith("process_fractions"):
                assert len(table["binned"].edges) == len(table["split"].edges) - 1
                continue
            for binned in (table["fake_factors"], *table["non_closures"], *([table["dr_sr"], *table["dr_sr_non_closures"]] if "dr_sr" in table else [])):
                assert len(binned.edges) == len(binned.fits) == len(table["split"].edges) - 1, (channel, name, binned.variable)
    assert TABLES["mt"]["QCD"]["fake_factors"].edges == ((30.0, 33.53, 38.0, 44.89, 58.39, 150.0),)
    assert TABLES["tt"]["QCD"]["non_closures"][2].fits[0].binwise_left == 3  # binwise#[2]: the bins 0 to 2


def test_the_lt_fraction_split_lies_between_two_and_three_jets():
    """U4: the categories <=2 and >=3 (jvoss's edges [-0.5, 1.5, 22.5] route n_jets = 2 to the >=3 fractions)."""
    for channel in TAU_CHANNELS:
        assert TABLES[channel]["process_fractions"]["split"].edges == (-0.5, 2.5, 22.5)


def test_other_channels_are_rejected():
    with pytest.raises(AnalysisError, match="supports"):
        build(config(("mt", "em"), "bbtautau_shapesmith.ff_measurement:build"))
