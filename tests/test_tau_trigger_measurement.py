"""The tau-leg trigger measurement of the tt embedding (E33; v7_tt_emb_trigger 12.2, 12.5): selection, pass
definition per era and sample kind, bins, processes, weights, variants, the ES of the embedded taus, the column
contract, in every era."""
import numpy as np
import pandas as pd
import pytest
from shapesmith import payloads
from shapesmith.expressions import mask
from shapesmith.measurements.tau_id_es.combine import Interval
from shapesmith.measurements.tau_id_es.grid import CATEGORIES
from shapesmith.measurements.tau_id_es.payload import correction_set
from shapesmith.model import AnalysisError
from shapesmith.skim import needed_columns
from shapesmith.validate import validate

from bbtautau_shapesmith import samples as samples_module, tau_trigger_measurement as module
from bbtautau_shapesmith.tau_trigger_measurement import MONITOR_FLAGS, TRIGGER_OBJECT, build, pass_cut
from tests.helpers import config, sample_types, tau_id_branches
from tests.run2 import ERAS, install, list_name

YEAR = {"2016preVFP": "2016", "2016postVFP": "2016", "2017": "2017", "2018": "2018"}
TRIGGER = "bbtautau_shapesmith.tau_trigger_measurement:build"


@pytest.fixture(scope="module")
def run2(tmp_path_factory):
    directory = tmp_path_factory.mktemp("run2_trigger")
    database = install(directory)
    patch = pytest.MonkeyPatch()
    patch.setattr(samples_module, "INVENTORY_DIR", directory)
    yield database
    patch.undo()


def trigger(database, era="2018", **switches):
    analysis = build(config(("mt",), TRIGGER, era=era, database=database, sample_lists=[list_name("tau_id", era)], **switches))
    validate(analysis)
    return analysis


@pytest.mark.parametrize("era", ERAS)
@pytest.mark.parametrize("wp_class", ["Medium", "VVVLoose_not_Medium"])
def test_the_measurement_builds_in_every_era_and_reads_only_produced_columns(run2, era, wp_class):
    analysis = trigger(run2, era, wp_class=wp_class)
    measurement = analysis.measurement
    assert analysis.name == f"tau_trigger_embedding_{era}" and measurement.name == "tau_trigger" and measurement.wp_class == wp_class
    assert len(measurement.bins) == 5 * 6 and {b.decay_modes for b in measurement.bins} == {(-1,), (0,), (1,), (10,), (11,)}
    assert [b.name for b in measurement.bins][:6] == ["DMall_PT40_45", "DMall_PT45_50", "DMall_PT50_60", "DMall_PT60_80", "DMall_PT80_120", "DMall_PT120"]
    assert measurement.signal == "EMB" and measurement.mc_process == "ZTT" and measurement.counting_regions == ("counting", "counting_same_sign")
    assert set(measurement.systematic_variants) == ({"signal_mc", "tag_pt", "mt30"} | ({"monitor"} if era != "2018" else set()))
    channel = analysis.channel("mt")
    assert set(MONITOR_FLAGS[YEAR[era]].values()) | set(TRIGGER_OBJECT.values()) <= tau_id_branches("mt", era, "data")  # 12.4, C13
    types = sample_types()
    for sample in channel.samples:  # the columns of the tau-ID production (P1) of the sample's type
        missing = needed_columns(channel, sample) - tau_id_branches("mt", era, types[sample.nick])
        assert not missing, (sample.nick, sorted(missing))
    assert channel.skim["tau_iso"] == "(id_tau_vsJet_VVVLoose_2 > 0.5)" and "os" not in channel.skim  # both classes and charges


def test_the_selection_of_tag_and_probe(run2):
    cuts = trigger(run2, "2018").channel("mt").cuts
    assert cuts["trg_selection"] == "(pt_1 > 28) & (pt_1 >= 25) & ((trg_single_mu27 > 0.5) | (trg_single_mu24 > 0.5))"
    assert cuts["tau_pt"] == "(pt_2 >= 40)" and cuts["tau_eta"] == "(abs(eta_2) < 2.1)" and cuts["muon_eta"] == "(abs(eta_1) < 2.1)"
    assert cuts["against_electron"] == "(id_tau_vsEle_VVLoose_2 > 0.5)" and cuts["against_muon"] == "(id_tau_vsMu_Tight_2 > 0.5)"
    assert cuts["mt_cut"] == "(mt_1 < 65)" and cuts["hem_veto"] == "(hem_veto < 0.5)" and cuts["tau_iso"] == "(id_tau_vsJet_Medium_2 > 0.5)"
    assert trigger(run2, "2016preVFP").channel("mt").cuts["trg_selection"].startswith("(pt_1 > 24) & (pt_1 >= 24) & ")
    assert trigger(run2, "2017").channel("mt").cuts["trg_selection"] == "(pt_1 > 29) & (pt_1 >= 29) & (trg_single_mu27 > 0.5)"
    anti = trigger(run2, "2018", wp_class="VVVLoose_not_Medium").channel("mt")
    assert anti.cuts["tau_iso"] == "((id_tau_vsJet_VVVLoose_2 > 0.5) & (id_tau_vsJet_Medium_2 < 0.5))"
    assert anti.process("ZL").selection.weights["tau_id"] == "1.0" and trigger(run2, "2018").channel("mt").process("ZL").selection.weights["tau_id"].count("id_wgt_tau_vsJet_Medium_2") == 1


def _frame(**columns):
    n = len(next(iter(columns.values())))
    base = {name: np.zeros(n) for flags in MONITOR_FLAGS.values() for name in flags.values()} | {"trgobj_pt_2": np.full(n, 50.0), "trgobj_l1pt_2": np.full(n, 40.0)}
    base |= {"is_data": np.zeros(n, dtype=bool), "is_mc": np.zeros(n, dtype=bool), "is_embedding": np.zeros(n, dtype=bool)}
    return pd.DataFrame({**base, **columns})


def test_the_pass_definition_mirrors_the_tt_legs_per_era_and_kind():
    combined_only = _frame(trg_mon_mu19tau32_medcombiso_2=np.ones(2), is_data=np.array([True, False]), is_mc=np.array([False, True]))
    assert mask(combined_only, [pass_cut("2016")]).tolist() == [True, False]  # the CombinedIso path in data only (E7, D9)
    low_object = _frame(trg_mon_mu19tau32_mediso_2=np.ones(2), trgobj_pt_2=np.array([34.0, 36.0]))
    assert mask(low_object, [pass_cut("2016")]).tolist() == [False, True] and mask(low_object, [pass_cut("2016", "monitor")]).tolist() == [True, True]
    medium = _frame(trg_mon_mu24tau35_mediumiso_tightid_2=np.ones(3), trgobj_pt_2=np.array([38.0, 45.0, 45.0]), trgobj_l1pt_2=np.array([40.0, 40.0, 30.0]))
    assert mask(medium, [pass_cut("2017")]).tolist() == [False, True, False] and mask(medium, [pass_cut("2017", "monitor")]).tolist() == [False, True, True]
    tight_id = _frame(trg_mon_mu24tau35_tightiso_tightid_2=np.ones(1), trgobj_pt_2=np.array([36.0]))
    assert mask(tight_id, [pass_cut("2017")]).tolist() == [True]  # Tight35_TightID without the 40 GeV object cut
    hps = _frame(trg_mon_mu24tau35_mediumiso_hps_2=np.ones(1), trgobj_l1pt_2=np.array([10.0]))
    assert mask(hps, [pass_cut("2018")]).tolist() == [True] and mask(medium, [pass_cut("2018")]).tolist() == [False, True, True]  # no L1 cut in 2018


def test_variants(run2):
    assert trigger(run2, "2018", variant="tag_pt").channel("mt").cuts["trg_selection"].startswith("(pt_1 > 31) & ")
    assert trigger(run2, "2018", variant="mt30").channel("mt").cuts["mt_cut"] == "(mt_1 < 30)"
    signal_mc = trigger(run2, "2018", variant="signal_mc")
    assert signal_mc.measurement.signal == "ZTT" and signal_mc.measurement.monitor_process is None and "EMB" not in {p.name for p in signal_mc.channel("mt").processes}
    assert signal_mc.channel("mt").process("ZTT").role == "background" and signal_mc.channel("mt").process("TTT").role == "background"
    assert "ZTT" in signal_mc.measurement.qcd_subtract and signal_mc.measurement.split_signal_shapes and not signal_mc.measurement.pseudo_data
    closure = trigger(run2, "2018", variant="closure").measurement
    assert closure.signal == "ZTT" and closure.pseudo_data and not closure.split_signal_shapes and "closure" not in closure.systematic_variants
    with pytest.raises(AnalysisError, match="the variant monitor applies to 2016"):
        trigger(run2, "2018", variant="monitor")
    monitor = trigger(run2, "2017", variant="monitor").channel("mt")
    assert "trgobj_l1pt_2" not in next(c.cut for c in monitor.categories if c.name == "DM0_PT40_45_pass")


def test_categories_bins_and_their_common_m_vis_edges(run2, monkeypatch):
    channel = trigger(run2, "2018").channel("mt")
    passing = next(c for c in channel.categories if c.name == "DM1_PT60_80_pass")
    failing = next(c for c in channel.categories if c.name == "DM1_PT60_80_fail")
    assert passing.cut.startswith("((tau_decaymode_2 == 1)) & (pt_2 >= 60) & (pt_2 < 80) & ") and failing.cut.endswith(f"& ~{pass_cut('2018')}")
    assert passing.variable.edges == failing.variable.edges and passing.variable.edges.cut == "((tau_decaymode_2 == 1)) & (pt_2 >= 60) & (pt_2 < 80)"
    assert next(c for c in channel.categories if c.name == "DMall_PT120_pass").cut.startswith("(tau_decaymode_2 >= 0) & (pt_2 >= 120) & ")
    merged = trigger(run2, "2018", wp_class="VVVLoose_not_Medium", dm1011=True).measurement
    assert {b.decay_modes for b in merged.bins} == {(-1,), (0,), (1,), (10, 11)}
    monkeypatch.setitem(module.TRIGGER_PT_EDGES, "2017", {"Medium": {"DM0": (40.0, 50.0, 80.0)}})
    bins = trigger(run2, "2017").measurement.bins
    assert [b.name for b in bins if b.decay_modes == (0,)] == ["DM0_PT40_50", "DM0_PT50_80", "DM0_PT80"]


def test_processes_and_the_fit_model(run2):
    analysis = trigger(run2, "2016postVFP")
    channel, measurement = analysis.channel("mt"), analysis.measurement
    roles = {p.name: p.role for p in channel.processes}
    assert roles == {"data": "data", "EMB": "background", "ZTT": "auxiliary", "ZL": "background", "ZJ": "background", "TTT": "auxiliary", "TTL": "background",
                     "TTJ": "background", "STL": "background", "STJ": "background", "VVL": "background", "VVJ": "background", "W": "background"}
    qcd = next(e for e in channel.estimators if getattr(e, "output", None) == "QCD")
    assert qcd.region == "same_sign" and set(qcd.subtract) == {name for name, role in roles.items() if role == "background"}
    assert measurement.normalisations["CMS_tauTrg_norm_ZJ"] == (("ZJ",), 1.2) and measurement.normalisations["CMS_tauTrg_norm_TT"] == (("TTL", "TTJ"), 1.1)
    assert measurement.fake_processes == ("ZJ", "TTJ", "STJ", "VVJ", "W", "QCD") and measurement.fake_efficiency == 0.1
    assert measurement.signal_template is None and measurement.signal_shapes == ("CMS_emb_ttbar_contamination_Run2016postVFP",)  # no ES payload given


def test_the_energy_scale_of_the_embedded_taus_from_the_payload(run2, tmp_path):
    results = {("Medium", "VVLoose"): {name: (Interval(1.0, 0.95, 1.05), Interval(-1.0 if name.startswith("DM0") else 2.0, -2.0, 3.0)) for name in CATEGORIES}}
    payload = payloads.write(correction_set(results, {}), tmp_path / "es.json.gz")
    analysis = trigger(run2, "2018", emb_es_payload=str(payload))
    channel, measurement = analysis.channel("mt"), analysis.measurement
    template = next(v for v in channel.variations if v.name == "emb_es")
    assert template.derived["pt_2"] == "pt_2 * ((tau_decaymode_2 == 0) * 0.99 + (tau_decaymode_2 == 1) * 1.02 + (tau_decaymode_2 == 10) * 1.02 + (tau_decaymode_2 == 11) * 1.02)"
    up = next(v for v in channel.variations if v.name == "CMS_tauTrg_emb_es_Run2018Up")
    assert "(tau_decaymode_2 == 0) * 1.03" in up.derived["pt_2"] and template.applies_to == ("embedding",) and template.regions == ("nominal",)
    assert measurement.signal_template == "emb_es" and measurement.signal_shapes == ("CMS_tauTrg_emb_es_Run2018", "CMS_emb_ttbar_contamination_Run2018")
    with pytest.raises(AnalysisError, match="does not exist"):
        trigger(run2, "2018", emb_es_payload=str(tmp_path / "missing.json.gz"))


def test_only_mt(run2):
    with pytest.raises(AnalysisError, match="channel mt only"):
        build(config(("mt", "mm"), TRIGGER, era="2018", database=run2, sample_lists=[list_name("tau_id", "2018")]))
