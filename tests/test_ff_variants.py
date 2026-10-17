"""The fake-factor measurement of v7 (spec E18, 6.8): the variant with W+jets fake factors, the tables per era with
their placeholder edges, the combined 2016 measurement, and the W+jets friend shifts of the tau analysis."""
import pytest
from shapesmith.measurements.fake_factors import Binned
from shapesmith.model import AnalysisError
from shapesmith.skim import needed_columns
from shapesmith.validate import validate

from bbtautau_shapesmith import cuts, ff_measurement, ff_tables, samples as samples_module
from bbtautau_shapesmith.analysis import build as build_tau
from bbtautau_shapesmith.constants import FF_SHIFTS_W, LUMI_PB, TAU_CHANNELS, lumi_pb
from tests.helpers import branches, config, sample_types
from tests.run2 import install, list_name

FF = "bbtautau_shapesmith.ff_measurement:build"
T_PARTS, L_PARTS, J_PARTS = ("ZTT", "TTT", "STT", "VVT"), ("ZL", "TTL", "STL", "VVL"), ("ZJ", "TTJ", "STJ", "VVJ")


@pytest.fixture(scope="module")
def run2(tmp_path_factory):
    directory = tmp_path_factory.mktemp("run2_ff")
    database = install(directory)
    patch = pytest.MonkeyPatch()
    patch.setattr(samples_module, "INVENTORY_DIR", directory)
    yield database
    patch.undo()


def measurement(database, era="2018", channels=TAU_CHANNELS, eras=None, **switches):
    lists = [list_name(kind, member) for member in (eras or (era,)) for kind in ("binned", "embedding")]
    switches = {"sample_lists": lists, "embedding": ["et", "mt"], **switches}
    analysis = ff_measurement.build(config(channels, FF, era=era, database=database, **switches))
    validate(analysis)
    return analysis


W = ["QCD", "Wjets", "ttbar"]


def region_cuts(channel, name):
    return {**channel.cuts, **channel.region(name).replace_cuts}


def test_the_w_variant_adds_w_fake_factors_in_et_and_mt(run2):
    analysis = measurement(run2, ff_processes=W)
    assert analysis.measurement.name == "fake_factors_w"
    for name in ("et", "mt"):
        (leg,) = analysis.measurement.legs[name]
        assert [p.name for p in leg.processes] == ["QCD", "Wjets", "ttbar"]
        wjets = leg.wjets
        assert wjets.target == "data" and "W" not in wjets.subtract and wjets.subtract == ("EMB",) + L_PARTS + J_PARTS
        assert wjets.qcd.regions == {"Wjets_sr_like": "Wjets_sr_like_ss", "Wjets_ar_like": "Wjets_ar_like_ss"} and "W" in wjets.qcd.subtract
        assert wjets.dr_sr.target == "W" and wjets.dr_sr.subtract == () and wjets.dr_sr.correction.variable == "mt_1"
        assert (wjets.dr_sr.sr_like, wjets.dr_sr.ar_like, wjets.dr_sr.sr, wjets.dr_sr.ar) == ("Wjets_sr_like", "Wjets_ar_like", "Wjets_sr", "Wjets_ar")
        assert leg.fractions.wjets == "W" and leg.fractions.subtract == ("EMB",) + L_PARTS + J_PARTS + ("W",)
        channel = analysis.channel(name)
        sr_like = region_cuts(channel, "Wjets_sr_like")
        assert sr_like["mt_cut"] == "(mt_1 > 70)" and sr_like["b_tagging"] == "(n_bjets == 0) & (n_jets >= 2)" and sr_like["os"] == channel.cuts["os"]
        assert sr_like["lepton_iso"] == "(iso_1 < 0.15)" and sr_like["tau_iso"] == cuts.isolated(2)
        assert region_cuts(channel, "Wjets_ar_like_ss")["os"] == cuts.SAME_SIGN and region_cuts(channel, "Wjets_ar_like_ss")["tau_iso"] == cuts.anti_isolated(2)
        assert region_cuts(channel, "Wjets_sr") == channel.cuts and region_cuts(channel, "Wjets_ar")["tau_iso"] == cuts.anti_isolated(2)
        for sample in channel.samples:
            missing = needed_columns(channel, sample) - branches(name, "2018", sample_types()[sample.nick])
            assert not missing, (sample.nick, sorted(missing))
    tt_legs = analysis.measurement.legs["tt"]
    assert all(leg.wjets is None and leg.fractions.wjets is None for leg in tt_legs) and not [r for r in analysis.channel("tt").regions if r.name.startswith("Wjets")]


def test_the_variants_and_their_placeholder_edges(run2):
    classic = measurement(run2)
    assert classic.measurement.name == "fake_factors" and classic.measurement.placeholder_edges is None
    assert all(leg.wjets is None for legs in classic.measurement.legs.values() for leg in legs)
    note = measurement(run2, ff_processes=W).measurement.placeholder_edges
    assert note.startswith("8 quantities have no edges of 2018") and "Wjets_fake_factors" in note  # the W+jets edges come from --suggest-binning
    assert measurement(run2, channels=("tt",), ff_processes=W, embedding=False).measurement.placeholder_edges is None  # tt has no W+jets
    with pytest.raises(AnalysisError, match="ff_processes must be one of"):
        measurement(run2, ff_processes=["QCD", "Wjets"])


def test_the_tables_of_the_other_eras(monkeypatch):
    tables, placeholders = ff_tables.tables("2017")
    qcd = tables["mt"]["QCD"]["fake_factors"]
    assert qcd.equipopulated.n_bins == (round(ff_tables.TABLES["mt"]["QCD"]["fake_factors"].equipopulated.n_bins[0] * 42070 / 59560),)
    assert qcd.edges == ff_tables.TABLES["mt"]["QCD"]["fake_factors"].edges and "mt QCD_fake_factors" in placeholders  # 2018 edges as placeholders
    assert "mt QCD_non_closure_tau_decaymode_2" not in placeholders  # binwise without equipopulated options: fixed edges
    assert ff_tables.tables("2016preVFP") == ff_tables.tables("2016postVFP") == ff_tables.tables("2016")  # one table for both halves
    assert ff_tables.PER_ERA_OVERRIDES["2016"]["tt"]["ttbar_fake_factors"] == (2, 2)  # 3 and 4 bins in 2018: at least 2
    edges = {"QCD_fake_factors": [[30.0, 35.0, 45.0, 150.0]]}
    monkeypatch.setitem(ff_tables.PER_ERA_EDGES, "2017", {"mt": edges})
    tables, placeholders = ff_tables.tables("2017")
    assert tables["mt"]["QCD"]["fake_factors"].edges == ((30.0, 35.0, 45.0, 150.0),) and "mt QCD_fake_factors" not in placeholders
    monkeypatch.setitem(ff_tables.PER_ERA_EDGES, "2017", {"mt": {"QCD_fake_factors": [[30.0, 150.0], [30.0, 150.0]]}})
    with pytest.raises(ValueError, match="QCD_fake_factors: 2 categories of edges, the table has 1"):
        ff_tables.tables("2017")


def test_every_table_quantity_has_its_suggest_binning_name():
    for channel, channel_tables in ff_tables.tables("2018", wjets=True)[0].items():
        for key, table in channel_tables.items():
            names = ff_tables.quantity_names(key, table)
            assert all(isinstance(b, Binned) for b in names.values())
            if not key.startswith("process_fractions"):
                assert f"{key}_fake_factors" in names


def test_2016_is_measured_on_both_halves_with_their_luminosity(run2):
    analysis = measurement(run2, era="2016", eras=("2016preVFP", "2016postVFP"))
    assert analysis.name == "hh_bbtautau_2016_v15_fake_factors" and analysis.era == "2016" and analysis.lumi_pb == lumi_pb("2016") == 36330.0
    mt = analysis.channel("mt")
    by_era = {}
    for sample in mt.samples:
        by_era.setdefault(sample.era, []).append(sample)
        if sample.kind == "mc":
            assert sample.lumi_scale == pytest.approx(LUMI_PB[sample.era] / 36330.0)
        else:
            assert sample.lumi_scale == 1.0
    assert set(by_era) == {"2016preVFP", "2016postVFP"}
    for half in by_era:  # exactly the samples of the run of the half
        assert sorted(s.nick for s in by_era[half]) == sorted(s.nick for s in measurement(run2, era=half, channels=("mt",), embedding=["mt"]).channel("mt").samples)
    assert {s.group for s in by_era["2016preVFP"] if s.kind == "data"} == {"data"}
    assert "hem_veto" not in mt.cuts and mt.cuts["trg_selection"] == cuts.baseline_cuts("mt", "2016preVFP")["trg_selection"]
    assert analysis.measurement.placeholder_edges.startswith("60 quantities have no edges of 2016")
    with pytest.raises(AnalysisError, match="no DY table for 2016"):
        measurement(run2, era="2016", eras=("2016preVFP", "2016postVFP"), dy_correction="njets_btag")


def test_a_combined_era_takes_only_its_halves_and_borrows_nothing(run2):
    with pytest.raises(AnalysisError, match="a 2017 sample in a 2016 \\(2016preVFP \\+ 2016postVFP\\) run"):
        measurement(run2, era="2016", eras=("2016preVFP", "2017"))
    with pytest.raises(AnalysisError, match="a run of the combined era 2016 borrows nothing"):
        ff_measurement.build(config(("mt",), FF, era="2016", database=run2, sample_lists=[list_name("binned", "2016postVFP"), list_name("borrowed", "2016postVFP")]))


def test_the_tau_analysis_shifts_the_w_keys_of_the_friend(run2):
    lists = [list_name("binned", "2018"), list_name("embedding", "2018")]
    analysis = build_tau(config(("et", "mt", "tt"), era="2018", database=run2, sample_lists=lists, jet_fakes="ff", embedding=["et", "mt", "tt"], crown_shifts=False, ff_processes=W))
    validate(analysis)
    for name in ("et", "mt"):
        names = {v.name for v in analysis.channel(name).variations}
        assert all(f"CMS_ff_{key}_{name}_2018Up" in names for key in FF_SHIFTS_W)
    assert not [v for v in analysis.channel("tt").variations if "Wjets" in v.name]
    classic = build_tau(config(("mt",), era="2018", database=run2, sample_lists=lists, jet_fakes="ff", embedding=["mt"], crown_shifts=False))
    assert not [v for v in classic.channel("mt").variations if "Wjets" in v.name]
