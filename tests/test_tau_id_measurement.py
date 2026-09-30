from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import uproot
from shapesmith.config import load_config
from shapesmith.expressions import evaluate
from shapesmith.measurements.tau_id_es import check
from shapesmith.measurements.tau_id_es.grid import CATEGORIES, factor
from shapesmith.estimates import run_estimates
from shapesmith.histogram import HistKey, Histogram, HistogramSet, is_part, part_of
from shapesmith.measurements.tau_id_es.synced import shapes_path, write_synced
from shapesmith.model import AnalysisError, ColumnVariation, DataMinus, TemplateShift, VariationSum, applies
from shapesmith.validate import validate

from bbtautau_shapesmith.tau_id_binning import M_VIS_EDGES
from bbtautau_shapesmith.tau_id_measurement import ES_GRID, build, es_variation
from bbtautau_shapesmith.tau_id_systematics import SUMMED_SHIFTS, UNUSED_SHIFTS, WEIGHT_SHIFTS, mc_variation_sums
from tests.helpers import DATABASE, REPO, config

SYNCED = Path("/work/jvoss/smhtt_ul_SFs_v15/output/shapes_synced/SFs_EMB_Run2_04_08_26__full")
WP_TAGS = {("Medium", "VVLoose"): "M_VVL", ("Medium", "Tight"): "M_T", ("Tight", "VVLoose"): "T_VVL", ("Tight", "Tight"): "T_T"}
# the predecessor's trigger weight (smhtt_ul config/shapes/process_selection.py, 2018 mt and mm)
TRIGGER_WEIGHT = "((pt_1 >= 25) & (pt_1 < 28)) * trg_wgt_single_mu24 + (pt_1 > 28) * trg_wgt_single_mu27"
# the CROWN shifts of sm_tau_id_measurement_config (feat-new-sm-bbtautau-config 6f58e65, generated code) that move a
# column of the mt channel, for every MC sample type; the recoil shifts exist for DY and W only
PRODUCED_SHIFTS = (
    {"CMS_HEM_2018", "CMS_eff_m_trigger_2018", "CMS_pileup_2018", "CMS_res_j_2018", "CMS_scale_e_2018", "CMS_res_e_2018", "CMS_scale_met_unclustered_energy_2018"}
    | {f"CMS_scale_j_{s}" for s in ("Absolute", "Absolute_2018", "BBEC1", "BBEC1_2018", "EC2", "EC2_2018", "FlavorQCD", "HF", "HF_2018", "RelativeBal", "RelativeSample_2018")}
    | {f"CMS_fake_t_DeepTau2018v2p5_VSe_DM{dm}_{region}_2018" for dm in (0, 1, 10, 11) for region in ("barrel", "endcap")}
    | {f"CMS_fake_t_DeepTau2018v2p5_VSmu_wheel{w}_2018" for w in range(1, 6)}
    | {f"CMS_scale_t_DeepTau2018v2p5_DM{dm}_{gen}_2018" for dm in (0, 1, 10, 11)
       for gen in ("pt20to40_genTau", "pt40to60_genTau", "pt60toInf_genTau", "genElectron_barrel", "genElectron_endcap")}
    | {f"CMS_scale_t_DeepTau2018v2p5_genMuon_wheel{w}_2018" for w in range(1, 6)}
)
RECOIL_SHIFTS = {"CMS_res_met_RecoilCalibration_2018", "CMS_scale_met_RecoilCalibration_2018"}


def analysis_for(**switches):
    analysis = build(config(channels=("mt", "mm"), analysis="bbtautau_shapesmith.tau_id_measurement:build", sample_lists=["sm2018_tau_id_measurement"], **switches))
    validate(analysis)
    return analysis


def test_mt_selection_is_the_predecessors():
    # smhtt_ul config/shapes/channel_selection.py, special "TauID_ES", 2018 mt; bbtautau names dilepton_veto, id_tau_vsMu_Tight_2
    assert analysis_for(vsjet_wp="Medium", vsele_wp="Tight").channel("mt").cuts == {
        "os": "((q_1 * q_2) < 0)",
        "tau_decay_mode": "((tau_decaymode_2 == 0) | (tau_decaymode_2 == 1) | (tau_decaymode_2 == 10) | (tau_decaymode_2 == 11))",
        "tau_eta": "(abs(eta_2) < 2.5)",
        "extraelec_veto": "(extraelec_veto < 0.5)",
        "extramuon_veto": "(extramuon_veto < 0.5)",
        "dilepton_veto": "(dilepton_veto < 0.5)",
        "against_muon": "(id_tau_vsMu_Tight_2 > 0.5)",
        "against_electron": "(id_tau_vsEle_Tight_2 > 0.5)",
        "tau_iso": "(id_tau_vsJet_Medium_2 > 0.5)",
        "muon_iso": "(iso_1 < 0.15)",
        "mt_cut": "(mt_1 < 65)",
        "trg_selection": "(pt_2 > 20) & (pt_1 >= 25) & ((trg_single_mu27 > 0.5) | (trg_single_mu24 > 0.5))",
    }


def test_mm_control_region_is_the_predecessors():
    channel = analysis_for().channel("mm")
    assert channel.cuts == {
        "os": "((q_1 * q_2) < 0)",
        "m_vis": "(m_vis > 70) & (m_vis < 110)",
        "muon_iso": "(iso_1 < 0.15) & (iso_2 < 0.15)",
        "trg_selection": "(pt_2 > 15) & (pt_1 >= 25) & ((trg_single_mu27 > 0.5) | (trg_single_mu24 > 0.5))",
    }
    assert [(c.name, c.cut, c.variable.edges) for c in channel.categories] == [("control_region", "(m_vis > 70) & (m_vis < 110)", (70.0, 110.0))]


def test_categories_are_the_predecessors():
    # smhtt_ul config/shapes/tauid_measurement_binning.py: pT >= 20 GeV, bins [20, 40) and [40, 200]
    channel = analysis_for().channel("mt")
    cuts = {c.name: c.cut for c in channel.categories}
    assert list(cuts) == list(CATEGORIES)
    assert cuts["DM0"] == "(tau_decaymode_2 == 0) & (pt_2 >= 20)"
    assert cuts["DM1_PT20_40"] == "(tau_decaymode_2 == 1) & (pt_2 >= 20) & (pt_2 < 40)"
    assert cuts["DM1011_PT40_200"] == "((tau_decaymode_2 == 10) | (tau_decaymode_2 == 11)) & (pt_2 >= 40) & (pt_2 <= 200)"
    assert all(c.variable.edges == M_VIS_EDGES["Tight", "VVLoose"][c.name] for c in channel.categories)


def test_processes_weights_and_estimates():
    channel = analysis_for(vsjet_wp="Tight", vsele_wp="VVLoose").channel("mt")
    roles = {p.name: p.role for p in channel.processes}
    assert roles == {"data": "data", "EMB": "background", "ZL": "background", "ZJ": "background", "TTT": "auxiliary", "TTL": "background", "TTJ": "background",
                     "STL": "background", "STJ": "background", "VVL": "background", "VVJ": "background", "W": "background"}
    emb = channel.process("EMB").selection
    assert emb.cuts == {"genmatch": "((gen_match_1 == 4) & (gen_match_2 == 5))"}
    assert emb.weights == {"emb_genweight": "emb_genweight", "emb_selection": "emb_idsel_wgt_1 * emb_idsel_wgt_2 * emb_triggersel_wgt",
                           "emb_id": "id_wgt_mu_1", "emb_iso": "iso_wgt_mu_1", "emb_trigger": TRIGGER_WEIGHT}  # no tau ID, vsEle or vsMu SF
    assert channel.process("TTJ").selection.weights == {
        "puweight": "puweight", "id": "id_wgt_mu_1", "iso": "iso_wgt_mu_1", "tau_id": "((gen_match_2 == 5) * id_wgt_tau_vsJet_Tight_2 + (gen_match_2 != 5))",
        "vs_mu": "id_wgt_tau_vsMu_Tight_2", "vs_ele": "id_wgt_tau_vsEle_VVLoose_2", "trigger": TRIGGER_WEIGHT, "top_pt": "topPtReweightWeight", "jet_fake": "1.0"}
    assert channel.process("ZL").selection.cuts == {"genmatch": "(~((gen_match_1 == 4) & (gen_match_2 == 5)) & ~(gen_match_2 == 6))"}
    qcd, *sums, contamination = channel.estimators
    assert qcd == DataMinus("QCD", "same_sign", ("EMB", "ZL", "ZJ", "TTL", "TTJ", "STL", "STJ", "VVL", "VVJ", "W"), clip_negative=True)
    assert tuple(sums) == mc_variation_sums()
    assert contamination == TemplateShift("CMS_emb_ttbar_contamination_Run2018", "EMB", "TTT", 0.1)
    mm = analysis_for().channel("mm")
    assert mm.process("MUEMB").selection.cuts == {"genmatch": "(gen_match_1 == 2) & (gen_match_2 == 2)"}
    assert mm.process("MUEMB").selection.weights["emb_id"] == "id_wgt_mu_1 * id_wgt_mu_2"
    assert mm.estimators == (DataMinus("QCD", "same_sign", ("MUEMB", "W"), clip_negative=True),)


def test_working_point_switches():
    for vsjet, vsele in WP_TAGS:
        analysis = analysis_for(vsjet_wp=vsjet, vsele_wp=vsele)
        channel = analysis.channel("mt")
        assert channel.cuts["tau_iso"] == f"(id_tau_vsJet_{vsjet}_2 > 0.5)" and channel.cuts["against_electron"] == f"(id_tau_vsEle_{vsele}_2 > 0.5)"
        assert (analysis.measurement.vsjet_wp, analysis.measurement.vsele_wp) == (vsjet, vsele)
        check(analysis, analysis.measurement)
        assert channel.skim["tau_iso"] == "(id_tau_vsJet_Medium_2 > 0.5)" and channel.skim["against_electron"] == "(id_tau_vsEle_VVLoose_2 > 0.5)"
    skims = {tuple(sorted(analysis_for(vsjet_wp=j, vsele_wp=e).channel("mt").skim.items())) for j, e in WP_TAGS}
    assert len(skims) == 1  # one skim serves every combination
    with pytest.raises(AnalysisError, match="vsjet_wp"):
        analysis_for(vsjet_wp="Loose")


def _events(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "pt_1": rng.uniform(25, 80, n), "eta_1": rng.uniform(-2.4, 2.4, n), "phi_1": rng.uniform(-np.pi, np.pi, n), "mass_1": np.full(n, 0.10566),
        "pt_2": rng.uniform(20, 120, n), "eta_2": rng.uniform(-2.3, 2.3, n), "phi_2": rng.uniform(-np.pi, np.pi, n), "mass_2": rng.uniform(0.13, 1.6, n),
        "met": rng.uniform(0, 100, n), "metphi": rng.uniform(-np.pi, np.pi, n),
    })
    return frame


def _four_vector(pt, eta, phi, mass):
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return np.sqrt(px**2 + py**2 + pz**2 + mass**2), px, py, pz


def _shifted(frame, s):
    """m_vis, MET and mt_1 with the tau four-momentum scaled by s, computed from four-vectors."""
    e1, x1, y1, z1 = _four_vector(frame.pt_1, frame.eta_1, frame.phi_1, frame.mass_1)
    e2, x2, y2, z2 = _four_vector(frame.pt_2 * s, frame.eta_2, frame.phi_2, frame.mass_2 * s)
    met_x = frame.met * np.cos(frame.metphi) + (1 - s) * frame.pt_2 * np.cos(frame.phi_2)
    met_y = frame.met * np.sin(frame.metphi) + (1 - s) * frame.pt_2 * np.sin(frame.phi_2)
    met, metphi = np.hypot(met_x, met_y), np.arctan2(met_y, met_x)
    return {
        "m_vis": np.sqrt((e1 + e2) ** 2 - (x1 + x2) ** 2 - (y1 + y2) ** 2 - (z1 + z2) ** 2),
        "met": met, "metphi": metphi, "mt_1": np.sqrt(2 * frame.pt_1 * met * (1 - np.cos(frame.phi_1 - metphi))),
    }


@pytest.mark.parametrize("shift", [-200, -2, 34, 200])
def test_es_grid_expressions_scale_the_tau_four_momentum(shift):
    frame, s = _events(), factor(shift)
    nominal = _shifted(frame, 1.0)
    frame["m_vis"], frame["mt_1"] = nominal["m_vis"], nominal["mt_1"]
    derived = {column: evaluate(frame, expr) for column, expr in es_variation(shift).derived.items()}
    expected = {**_shifted(frame, s), "pt_2": frame.pt_2 * s, "mass_2": frame.mass_2 * s}
    assert set(derived) == set(expected)
    for column, values in expected.items():
        assert derived[column] == pytest.approx(np.asarray(values), rel=1e-9, abs=1e-9), column


def test_es_grid_is_embedding_only_nominal_only():
    channel = analysis_for().channel("mt")
    grid = [v for v in channel.variations if isinstance(v, ColumnVariation) and v.derived]
    assert [v.name for v in grid] == [f"es{shift:+d}" for shift in ES_GRID] and len(grid) == 200
    assert all(v.applies_to == ("embedding",) and v.regions == ("nominal",) for v in grid)
    assert not analysis_for().channel("mm").variations


def test_mc_systematics_are_the_predecessor_families():
    # the shape uncertainties the patched HttSystematics_TauIDRun2.cc declares for mt with embedding (the regrouped JES
    # and HEM), plus CMS_PileUp of the synced shapes
    expected = {f"CMS_scale_j_{s}_Run2018" for s in ("Absolute", "BBEC1", "EC2", "HF", "RelativeSample", "HEMIssue")}
    expected |= {f"CMS_scale_j_{s}" for s in ("Absolute", "BBEC1", "EC2", "HF", "FlavorQCD", "RelativeBal")}
    expected |= {f"CMS_{name}_Run2018" for name in ("eff_m_trigger", "scale_met_unclustered_energy", "scale_met", "res_met", "scale_fake_m", "fake_j")}
    expected |= {f"CMS_{kind}_t_dm{dm}_Run2018" for kind in ("scale", "eff") for dm in (0, 1, 10, 11)} | {f"CMS_fake_m_WH{w}_Run2018" for w in range(1, 6)}
    expected |= {"CMS_htt_ttbarShape", "CMS_PileUp"}
    summed = {f"CMS_scale_t_dm{dm}_Run2018" for dm in (0, 1, 10, 11)} | {"CMS_scale_fake_m_Run2018"}
    channel = analysis_for(vsjet_wp="Medium").channel("mt")
    sums = {e.name: e.parts for e in channel.estimators if isinstance(e, VariationSum)}
    assert set(sums) == summed
    assert sums["CMS_scale_t_dm1_Run2018"] == ("pt20to40", "pt40to60", "pt60toInf") and sums["CMS_scale_fake_m_Run2018"] == tuple(f"wheel{w}" for w in range(1, 6))
    assert set(SUMMED_SHIFTS["CMS_scale_t_dm1_Run2018"].values()) == {f"CMS_scale_t_DeepTau2018v2p5_DM1_{pt}_genTau_2018" for pt in sums["CMS_scale_t_dm1_Run2018"]}  # the predecessor used the DM0 shift
    direct = {v.name for v in channel.variations if not (isinstance(v, ColumnVariation) and v.derived) and not is_part(v.name)}
    assert direct == {f"{name}{d}" for name in (expected - summed) | set(UNUSED_SHIFTS) for d in ("Up", "Down")}
    parts = {v.name for v in channel.variations if is_part(v.name)}
    assert parts == {f"{name}{d}%{part}" for name, family in sums.items() for part in family for d in ("Up", "Down")}
    no_op = next(v for v in channel.variations if v.name == "CMS_eff_t_dm1_Run2018Up")  # as the predecessor's: equal to the nominal
    assert no_op.replace_weights == {"tau_id": channel.process("TTL").selection.weights["tau_id"]}
    assert {name for name, *_ in WEIGHT_SHIFTS} == {"CMS_fake_j_Run2018", "CMS_htt_ttbarShape"}
    without = analysis_for(shape_systematics=False).channel("mt")
    assert not {v.name for v in without.variations} - {f"es{shift:+d}" for shift in ES_GRID}
    assert not [e for e in without.estimators if isinstance(e, VariationSum)]


def test_every_produced_shift_is_declared():
    # the skim fails on a shifted branch of a read column that no variation declares, and on a declared one without branch
    channel = analysis_for().channel("mt")
    for group in ("DY", "W", "TT", "ST", "VV"):
        suffixes = {v.suffix for v in channel.variations if isinstance(v, ColumnVariation) and v.suffix and applies(v, "mc", group)}
        shifts = PRODUCED_SHIFTS | (RECOIL_SHIFTS if group in ("DY", "W") else set())
        assert suffixes == {f"__{shift}{d}" for shift in shifts for d in ("Up", "Down")}, group
    assert not [v for v in channel.variations if isinstance(v, ColumnVariation) and v.suffix and applies(v, "embedding", "EMB")]


def test_summed_families_reach_the_shapes_file(tmp_path):
    analysis = analysis_for()
    hset, edges = HistogramSet(), list(M_VIS_EDGES["Tight", "VVLoose"]["DM0"])
    n = len(edges) - 1

    def hist(value):
        return Histogram(edges, np.full(n, value), np.full(n, 0.1))

    key = HistKey("mt", "DM0", "ZL", "nominal", "Nominal", "m_vis")
    hset[key] = hist(10.0)
    for d, sign in (("Up", 1.0), ("Down", -1.0)):
        for part, delta in zip(("pt20to40", "pt40to60", "pt60toInf"), (1.0, 0.5, 0.25)):
            hset[HistKey("mt", "DM0", "ZL", "nominal", part_of(f"CMS_scale_t_dm0_Run2018{d}", part), "m_vis")] = hist(10.0 + sign * delta)
    run_estimates(hset, analysis, ["mt"])
    write_synced(hset, analysis, tmp_path)
    with uproot.open(shapes_path(tmp_path, "mt", "2018")) as f:
        names = f.keys(recursive=True, cycle=False)
        assert f["mt_DM0/ZL_CMS_scale_t_dm0_Run2018Up"].values().tolist() == [11.75] * n
        assert f["mt_DM0/ZL_CMS_scale_t_dm0_Run2018Down"].values().tolist() == [8.25] * n
        assert not [name for name in names if "%" in name]


def test_the_run_configuration_builds():
    run = load_config(REPO / "configs" / "tau_id_es_2018.yaml", [f"sample_database={DATABASE}"])
    analysis = build(run)
    validate(analysis)
    assert analysis.measurement.name == "tau_id_es" and run.combine.cmssw_dir.endswith("CMSSW_14_1_0_pre4")


@pytest.mark.skipif(not SYNCED.exists(), reason="needs the predecessor's synced shapes")
@pytest.mark.parametrize("wps", list(WP_TAGS), ids=list(WP_TAGS.values()))
def test_bin_edges_are_those_of_the_predecessor_shapes(wps):
    import uproot

    path = SYNCED / f"2018-{WP_TAGS[wps]}_18_full" / wps[0] / wps[1] / "mt" / "htt_mt.inputs-sm-Run2018-TauID_ES.root"
    with uproot.open(path) as f:
        for category, edges in M_VIS_EDGES[wps].items():
            assert tuple(f[f"mt_{category}/data_obs"].axis().edges()) == edges
