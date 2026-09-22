import pytest
from shapesmith.config import NtupleConfig, RunConfig, load_analysis, load_config
from shapesmith.model import AnalysisError
from shapesmith.skim import required_columns
from typer.testing import CliRunner

from bbtautau_shapesmith.analysis import build
from bbtautau_shapesmith.constants import TAU_CHANNELS, FF_COLUMN_SET, NN_COLUMNS
from bbtautau_shapesmith.systematics import lnn, btag_variations
from bbtautau_shapesmith.processes import tau_processes
from tests.helpers import REPO, branches, nicks


def _config(**switches):
    return RunConfig(analysis="bbtautau_shapesmith.analysis:build", era="2018", channels=list(TAU_CHANNELS), switches=switches, ntuples=NtupleConfig(base="/x"), skim_dir="/s", output_dir="/o", sample_database=REPO / "tests" / "fixtures" / "datasets.json")


def test_mc_mode_without_nn_friend_validates():
    analysis = build(_config(jet_fakes="mc"))
    analysis.validate()
    assert analysis.categories == () and analysis.estimator.name == "abcd" and analysis.estimator.output == "QCD"
    assert "ZJ" in analysis.backgrounds() and "W" in analysis.backgrounds() and "QCD" in analysis.backgrounds()
    assert set(analysis.estimator.subtract) == {"ZTT", "TTT", "STT", "VVT", "TTVT", "ZL", "TTL", "STL", "VVL", "TTVL", "ZJ", "TTJ", "STJ", "VVJ", "TTVJ", "W"}
    assert len(analysis.samples) == 79 and analysis.lumi_pb == 59830.0


@pytest.mark.parametrize("sample_list", ["sm2018_binned_v1", "sm2018_binned_v2"])
def test_sample_list_selects_inventory(sample_list):
    analysis = build(_config(sample_list=sample_list))
    analysis.validate()
    assert {sample.nick for sample in analysis.samples} == set(nicks(sample_list))


def test_missing_sample_list_is_reported():
    with pytest.raises(FileNotFoundError, match="missing_sample_list"):
        build(_config(sample_list="missing_sample_list"))


@pytest.mark.parametrize("switches", [
    {"production": "sm2018_binned_v1"},
    {"production": "sm2018_binned_v1", "sample_list": "sm2018_binned_v2"},
])
def test_renamed_switch_is_reported_instead_of_using_default_samples(switches):
    with pytest.raises(AnalysisError, match=r"production.*sample_list"):
        build(_config(**switches))


def test_ff_mode_with_nn_friend_validates():
    analysis = build(_config(jet_fakes="ff", nn_friend=True))
    analysis.validate()
    assert len(analysis.categories) == 7 and analysis.estimator.name == "fake_factors" and analysis.estimator.output == "jetFakes"
    assert set(analysis.estimator.subtract) == {"ZTT", "TTT", "STT", "VVT", "TTVT", "ZL", "TTL", "STL", "VVL", "TTVL"}
    assert analysis.ml.region_of == {"jetFakes": "anti_iso"} and "jetFakes" in analysis.ml.processes


def test_embedding_needs_samples():
    with pytest.raises(AnalysisError, match="EMB"):
        build(_config(embedding=True)).validate()


@pytest.mark.parametrize("switches", [dict(jet_fakes="mc"), dict(jet_fakes="ff", nn_friend=True)])
@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_all_required_columns_exist(switches, channel):
    analysis = build(_config(**switches))
    allowed = branches(channel) | (NN_COLUMNS if switches.get("nn_friend") else set()) | (FF_COLUMN_SET if switches.get("jet_fakes") == "ff" else set())
    assert required_columns(analysis, channel) <= allowed, required_columns(analysis, channel) - allowed


def test_systematics():
    variations = btag_variations()
    assert len(variations) == 80 and variations[0].name == "CMS_btag_as_2018Up" and variations[0].replace_weights == {"btag": "btag_weight_upart_up_as"}
    entries = lnn(tau_processes("mc", False), "QCD")
    names = [e.name for e in entries]
    assert "lumi_13TeV_$ERA" in names and "QCDNorm_$CHANNEL_$ERA" in names and "htt_wjXsec" in names and "QCDscale_HH" in names
    assert "htt_wjXsec" not in [e.name for e in lnn(tau_processes("ff", False), "jetFakes")]
    assert all(e.processes for e in entries)


def test_style_covers_all_groups():
    analysis = build(_config(jet_fakes="mc"))
    groups = {p.plot_group for p in analysis.processes if p.kind not in ("data", "signal")} | {"QCD", "jetFakes", "EMB", analysis.signal}
    assert groups <= set(analysis.style.colors)
    assert set(analysis.style.group_order) >= groups - {analysis.signal}
    for variable in analysis.control_variables:
        assert variable in analysis.style.axis_labels["mt"]


@pytest.mark.parametrize("filename", ["sm2018_binned_v2.yaml", "sm2018_binned_v4.yaml", "sm2018_binned_v4_dilepton.yaml"])
def test_run_config_file_loads_and_validates(filename):
    config = load_config(REPO / "configs" / filename)
    assert config.switches["sample_list"] == "sm2018_binned_v2"
    if not config.sample_database.exists():
        pytest.skip("sample database checkout ../KingMaker_sample_database not available")
    load_analysis(config)
    from shapesmith.cli import app

    result = CliRunner().invoke(app, ["validate", "-c", str(REPO / "configs" / filename)])
    assert result.exit_code == 0, result.output
