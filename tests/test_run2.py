"""Run 2 (v7 spec 6.1-6.7): every analysis in every era, on the v7 inventories (tests/run2.py) and the per-era branch
fixtures (tests/helpers.py)."""
import numpy as np
import pandas as pd
import pytest
from shapesmith.datacards import write_datacard
from shapesmith.expressions import columns_of, mask, product
from shapesmith.histogram import HistKey, Histogram, HistogramSet
from shapesmith.model import AnalysisError, ColumnVariation, LnN, TemplateShift, WeightVariation
from shapesmith.skim import needed_columns
from shapesmith.validate import validate

from bbtautau_shapesmith import dilepton, ff_measurement, samples as samples_module, tau_id_measurement, weights as weights_module
from bbtautau_shapesmith.analysis import build
from bbtautau_shapesmith.constants import DILEPTON_CHANNELS, DY_NJETS_BTAG, FF_COLUMNS, LUMI_PB, NN_COLUMNS, TAU_CHANNELS
from bbtautau_shapesmith.correlations import FAMILIES, era_token, nuisance
from bbtautau_shapesmith.processes import NODE_SM, SIGNAL, VH_TAUTAU
from bbtautau_shapesmith.samples import parse_entry, route, samples
from bbtautau_shapesmith.style import lumi_label
from bbtautau_shapesmith.systematics import btag_variations
from bbtautau_shapesmith.tau_id_systematics import crown_shifts, unused_shifts, weight_shifts
from bbtautau_shapesmith.triggers import TRIGGER
from bbtautau_shapesmith.weights import embedding_weights, mc_weights
from tests.helpers import DATABASE, branches, config, embedding_branches, sample_types, tau_id_branches
from tests.run2 import ERAS, POWHEG, VH_TAUTAU as VH_PRIMARIES, install, list_name, nick, sample_lists

YEAR = {"2016preVFP": "2016", "2016postVFP": "2016", "2017": "2017", "2018": "2018"}
DILEPTON = "bbtautau_shapesmith.dilepton:build"
FF = "bbtautau_shapesmith.ff_measurement:build"
TAU_ID = "bbtautau_shapesmith.tau_id_measurement:build"
MU_2016 = "((trg_single_mu22 > 0.5) | (trg_single_mu22_tk > 0.5) | (trg_single_mu22_eta2p1 > 0.5) | (trg_single_mu22_eta2p1_tk > 0.5))"
TT_2018 = "((trg_double_tau35_mediumiso > 0.5) | (trg_double_tau35_tightiso > 0.5) | (trg_double_tau40_mediumiso > 0.5) | (trg_double_tau40_tightiso > 0.5))"
# spec 6.2, the trigger cut (with the offline thresholds) of the tau channels per year
TRIGGER_CUTS = {
    "2016": {
        "mt": f"(pt_2 > 30) & (pt_1 > 24) & {MU_2016}",
        "et": "(pt_2 > 30) & (pt_1 > 27) & (trg_single_ele25 > 0.5)",
        "tt": "((is_embedding > 0.5) | ((is_data > 0.5) & ((trg_double_tau35_mediumiso > 0.5) | (trg_double_tau35_mediumcombinediso > 0.5)))"
              " | ((is_mc > 0.5) & (trg_double_tau35_mediumiso > 0.5)))",
    },
    "2017": {
        "mt": "(pt_2 > 30) & (pt_1 > 29) & (trg_single_mu27 > 0.5)",
        "et": "(pt_2 > 30) & (((pt_1 >= 34) & (pt_1 < 37) & (trg_single_ele32 > 0.5)) | ((pt_1 >= 37) & (trg_single_ele35 > 0.5)))",
        "tt": "((is_embedding > 0.5) | ((trg_double_tau35_tightiso > 0.5) | (trg_double_tau40_mediumiso > 0.5) | (trg_double_tau40_tightiso > 0.5)))",
    },
    "2018": {
        "mt": "(pt_2 > 30) & (pt_1 > 25) & (trg_single_mu24 > 0.5)",
        "et": "(pt_2 > 30) & (pt_1 > 33) & (trg_single_ele32 > 0.5)",
        "tt": f"((is_embedding > 0.5) | {TT_2018})",
    },
}
MUON_WEIGHT = {"2016": "trg_wgt_single_mu22", "2017": "trg_wgt_single_mu27", "2018": "trg_wgt_single_mu24"}
LUMI = {  # spec 6.6
    "2016": {"lumi_13TeV_1516_l": 1.0118, "lumi_13TeV_151617_l": 1.0004, "lumi_13TeV_15161718_l": 1.0035},
    "2017": {"lumi_13TeV_151617_l": 1.0055, "lumi_13TeV_15161718_l": 1.0061},
    "2018": {"lumi_13TeV_15161718_l": 1.0084},
}


@pytest.fixture(scope="module")
def run2(tmp_path_factory):
    """Copies of the inventories (a test adds a list there); the fixture database of their nicks."""
    directory = tmp_path_factory.mktemp("run2")
    database = install(directory)
    patch = pytest.MonkeyPatch()
    patch.setattr(samples_module, "INVENTORY_DIR", directory)
    yield database
    patch.undo()


def tau(database, era, channels=("et", "mt", "tt", "em"), **switches):
    """The tau analysis of an era with embedding in every tau channel, fake factors and a nominal-only production."""
    switches = {"sample_lists": sample_lists(era, "binned", "embedding"), "jet_fakes": "ff", "embedding": ["et", "mt", "tt"], "crown_shifts": False, **switches}
    analysis = build(config(channels, era=era, database=database, **switches))
    validate(analysis)
    return analysis


def controls(database, era, **switches):
    analysis = dilepton.build(config(DILEPTON_CHANNELS, DILEPTON, era=era, database=database, sample_lists=sample_lists(era, "binned"), **switches))
    validate(analysis)
    return analysis


def fake_factors(database, era):
    analysis = ff_measurement.build(config(TAU_CHANNELS, FF, era=era, database=database, sample_lists=sample_lists(era, "binned", "embedding"), embedding=["et", "mt"]))
    validate(analysis)
    return analysis


def tau_id(database, era, **switches):
    analysis = tau_id_measurement.build(config(("mt", "mm"), TAU_ID, era=era, database=database, sample_lists=[list_name("tau_id", era)], **switches))
    validate(analysis)
    return analysis


@pytest.mark.parametrize("era", ERAS)
def test_every_analysis_builds_in_every_era(run2, era):
    analyses = (tau(run2, era), controls(run2, era), fake_factors(run2, era), tau_id(run2, era))
    assert [a.name for a in analyses] == [f"hh_bbtautau_{era}_v15", f"hh_bbtautau_{era}_dilepton_controls", f"hh_bbtautau_{era}_v15_fake_factors", f"tau_id_es_embedding_{era}"]
    assert all(a.era == era and a.lumi_pb == LUMI_PB[era] for a in analyses)
    assert analyses[0].style.lumi_label == analyses[1].style.lumi_label == lumi_label(era)


@pytest.mark.parametrize("analysis", ["bbtautau_shapesmith.analysis:build", DILEPTON, FF, TAU_ID])
def test_an_unknown_era_is_rejected(run2, analysis):
    """Only the fake-factor measurement takes the combined era 2016 (both halves, E18)."""
    module, _, function = analysis.partition(":")
    builder = getattr(__import__(module, fromlist=[function]), function)
    expected = "the measurement supports 2016preVFP, 2016postVFP, 2017, 2018 and 2016 \\(2016preVFP \\+ 2016postVFP\\)" if analysis == FF else "the analysis supports 2016preVFP, 2016postVFP, 2017, 2018"
    for era in ("2022", "Run2") if analysis == FF else ("2022", "2016", "Run2"):
        with pytest.raises(AnalysisError, match=f"unknown era '{era}'; {expected}"):
            builder(config(("mt",), analysis, era=era, database=run2))


def test_the_luminosity_and_its_label():
    assert LUMI_PB == {"2016preVFP": 19520.0, "2016postVFP": 16810.0, "2017": 42070.0, "2018": 59560.0}  # E20
    assert [lumi_label(era) for era in ERAS] == [
        r"19.5 fb$^{-1}$ (2016preVFP, 13 TeV)", r"16.8 fb$^{-1}$ (2016postVFP, 13 TeV)", r"42.1 fb$^{-1}$ (2017, 13 TeV)", r"59.6 fb$^{-1}$ (2018, 13 TeV)"]
    assert lumi_label("Run2") == r"138 fb$^{-1}$ (13 TeV)" and round(sum(LUMI_PB.values()) / 1000) == 138


@pytest.mark.parametrize("era", ERAS)
def test_every_needed_column_is_in_the_v7_contract(run2, era):
    """The columns of every sample of the tau, dilepton and fake-factor analyses are those CROWN writes in v7 for the
    sample type of the sample."""
    friends = NN_COLUMNS | set(FF_COLUMNS.values())
    types = sample_types()
    for analysis in (tau(run2, era, nn_friend=True), controls(run2, era, diagnostics=True), fake_factors(run2, era)):
        for name, channel in analysis.channels.items():
            for sample in channel.samples:
                missing = needed_columns(channel, sample) - branches(name, era, types[sample.nick]) - friends
                assert not missing, (analysis.name, name, sample.nick, sorted(missing))


@pytest.mark.parametrize("era", ERAS)
def test_the_tau_id_measurement_reads_the_columns_of_the_tau_id_production(run2, era):
    """The nominal columns of every sample of the tau-ID measurement are those of the tau-ID production (P1) of its
    sample type; the shifted ones come with the MC shifts of P1 (E34)."""
    types = sample_types()
    for vsjet, vsele in (("Medium", "VVLoose"), ("Tight", "Tight")):
        analysis = tau_id(run2, era, vsjet_wp=vsjet, vsele_wp=vsele, shape_systematics=False)
        for name, channel in analysis.channels.items():
            for sample in channel.samples:
                missing = needed_columns(channel, sample) - tau_id_branches(name, era, types[sample.nick])
                assert not missing, (name, sample.nick, sorted(missing))


@pytest.mark.parametrize("era", ERAS)
def test_the_trigger_table(run2, era):
    year = YEAR[era]
    analysis = tau(run2, era)
    for channel, expected in TRIGGER_CUTS[year].items():
        assert TRIGGER[era][channel].cut == analysis.channel(channel).cuts["trg_selection"] == analysis.channel(channel).skim["trg_selection"] == expected
    mt, et = analysis.channel("mt"), analysis.channel("et")
    assert mt.process("ZL").selection.weights["trigger"] == MUON_WEIGHT[year] == mt.process("EMB").selection.weights["emb_trigger"]
    ele = {"2016": "trg_wgt_single_ele25", "2017": "((pt_1 < 37) * trg_wgt_single_ele32 + (pt_1 >= 37) * trg_wgt_single_ele35)", "2018": "trg_wgt_single_ele32"}[year]
    assert et.process("ZL").selection.weights["trigger"] == ele == et.process("EMB").selection.weights["emb_trigger"]
    tm = tau_id(run2, era).channel("mt")
    assert tm.cuts["trg_selection"] == {"2016": f"(pt_2 > 20) & (pt_1 >= 24) & {MU_2016}", "2017": "(pt_2 > 20) & (pt_1 >= 29) & (trg_single_mu27 > 0.5)",
                                        "2018": "(pt_2 > 20) & (pt_1 >= 25) & ((trg_single_mu27 > 0.5) | (trg_single_mu24 > 0.5))"}[year]
    assert tm.process("TTJ").selection.weights["trigger"] == tm.process("EMB").selection.weights["emb_trigger"] == TRIGGER[era]["tau_id"].mc_weight
    if year != "2018":
        assert TRIGGER[era]["tau_id"].mc_weight == MUON_WEIGHT[year]


@pytest.mark.parametrize("era", ERAS)
def test_the_dilepton_triggers(run2, era):
    year = YEAR[era]
    channels = controls(run2, era).channels
    expected = {
        "2016": {"em": ("(pt_1 > 15) & (pt_2 > 24)", f"(abs(eta_2) < 2.1) & {MU_2016}"), "mm": ("(pt_1 > 24) & (pt_2 > 15)", f"(abs(eta_1) < 2.1) & {MU_2016}"),
                 "ee": ("(pt_1 > 27) & (pt_2 > 15)", "(abs(eta_1) < 2.1) & (trg_single_ele25 > 0.5)")},
        "2017": {"em": ("(pt_1 > 15) & (pt_2 > 29)", "(trg_single_mu27 > 0.5)"), "mm": ("(pt_1 > 29) & (pt_2 > 15)", "(trg_single_mu27 > 0.5)"),
                 "ee": ("(pt_1 >= 34) & (pt_2 > 15)", "(((pt_1 >= 34) & (pt_1 < 37) & (trg_single_ele32 > 0.5)) | ((pt_1 >= 37) & (trg_single_ele35 > 0.5)))")},
        "2018": {"em": ("(pt_1 > 15) & (pt_2 > 26)", "trg_single_mu24 > 0.5"), "mm": ("(pt_1 > 26) & (pt_2 > 15)", "trg_single_mu24 > 0.5"),
                 "ee": ("(pt_1 > 34) & (pt_2 > 15)", "trg_single_ele32 > 0.5")},
    }[year]
    for name, (thresholds, trigger) in expected.items():
        assert (channels[name].cuts["pt_selection"], channels[name].cuts["trigger"]) == (thresholds, trigger), name
        weight = channels[name].process("DY").selection.weights["trigger"]
        assert weight == (MUON_WEIGHT[year] if name != "ee" else TRIGGER[era]["et"].mc_weight)
        assert ("zvtx" in channels[name].process("DY").selection.weights) == (year == "2017" and name == "ee")


def test_the_2017_electron_trigger_split_and_the_zvtx_factor(run2):
    """E6: Ele32 for 34 <= pT < 37 GeV, Ele35 from 37 GeV; the weight piecewise, times 0.991 above 37 GeV, on MC and
    embedding."""
    et = tau(run2, "2017").channel("et")
    frame = pd.DataFrame({"pt_2": 40.0, "pt_1": [33.9, 35.0, 35.0, 38.0, 38.0], "trg_single_ele32": [1, 1, 0, 1, 0], "trg_single_ele35": [1, 0, 1, 0, 1],
                          "trg_wgt_single_ele32": 0.9, "trg_wgt_single_ele35": 0.8})
    assert mask(frame, [et.cuts["trg_selection"]]).tolist() == [False, True, False, False, True]
    for process, prefix in (("ZL", ""), ("EMB", "emb_")):
        weights = et.process(process).selection.weights
        assert product(frame, [weights[f"{prefix}trigger"], weights[f"{prefix}zvtx"]]).tolist() == pytest.approx([0.9, 0.9, 0.9, 0.8 * 0.991, 0.8 * 0.991])
    for era in ("2016preVFP", "2016postVFP", "2018"):
        assert not any("zvtx" in p.selection.weights or "emb_zvtx" in p.selection.weights for p in tau(run2, era).channel("et").processes)


@pytest.mark.parametrize("era", ERAS)
def test_the_tt_trigger_by_sample_kind(run2, era):
    """E7, E33: the embedded sample without trigger requirement; in 2016 the data take Iso or CombinedIso, the MC Iso."""
    tt = tau(run2, era).channel("tt")
    flags = {"2016": ("trg_double_tau35_mediumiso", "trg_double_tau35_mediumcombinediso"), "2017": ("trg_double_tau35_tightiso", "trg_double_tau40_mediumiso"),
             "2018": ("trg_double_tau35_mediumiso", "trg_double_tau40_tightiso")}[YEAR[era]]
    rows = []
    for kind in ("data", "mc", "embedding"):
        for fired in ((0, 0), (1, 0), (0, 1)):
            row = dict.fromkeys(columns_of([tt.cuts["trg_selection"]]), 0)
            rows.append(row | {"is_data": kind == "data", "is_mc": kind == "mc", "is_embedding": kind == "embedding"} | dict(zip(flags, fired)))
    passed = mask(pd.DataFrame(rows), [tt.cuts["trg_selection"]]).tolist()
    mc = [False, True, False] if YEAR[era] == "2016" else [False, True, True]  # 2016 MC: the Iso path only (D9)
    assert passed == [False, True, True] + mc + [True, True, True]
    if YEAR[era] == "2016":  # the MC weight is the SF of the Iso path
        weight = tt.process("ZL").selection.weights["trigger"]
        assert columns_of([weight]) == {"trg_double_tau35_mediumiso", "trg_wgtdouble_tau35_mediumiso_leg1", "trg_wgtdouble_tau35_mediumiso_leg2"}
        frame = pd.DataFrame({"trg_double_tau35_mediumiso": [1, 0], "trg_wgtdouble_tau35_mediumiso_leg1": 0.9, "trg_wgtdouble_tau35_mediumiso_leg2": 0.8})
        assert product(frame, [weight]).tolist() == pytest.approx([0.72, 0.0])


def test_the_tt_embedding_trigger_weight_follows_the_isolation_of_each_leg(run2):
    """E33: eps_data(tau1) * eps_data(tau2), Medium for an isolated leg, AntiMedium (VVVLoose and not Medium) else."""
    weight = embedding_weights("tt", "2018")["emb_trigger"]
    frame = pd.DataFrame({"id_tau_vsJet_Medium_1": [1, 1, 0, 0], "id_tau_vsJet_Medium_2": [1, 0, 1, 0],
                          "trg_effdata_emb_Medium_1": 0.8, "trg_effdata_emb_AntiMedium_1": 0.6, "trg_effdata_emb_Medium_2": 0.7, "trg_effdata_emb_AntiMedium_2": 0.5})
    assert product(frame, [weight]).tolist() == pytest.approx([0.56, 0.4, 0.42, 0.3])
    tt = tau(run2, "2018").channel("tt")
    assert tt.process("EMB").selection.weights["emb_trigger"] == weight
    assert "emb_trigger" not in tt.region("anti_iso").replace_weights  # the AR keeps the per-leg weight


@pytest.mark.parametrize("era", ERAS)
def test_hem_veto_in_2018_and_the_lepton_eta_in_every_era(run2, era):
    hem = {"hem_veto": "(hem_veto < 0.5)"} if era == "2018" else {}
    selections = []
    for analysis in (tau(run2, era), controls(run2, era), fake_factors(run2, era), tau_id(run2, era)):
        for name, channel in analysis.channels.items():
            selections.append((analysis.name, name, channel.cuts, channel.skim))
    for analysis, name, cuts, skim in selections:
        assert {k: v for k, v in cuts.items() if k == "hem_veto"} == {k: v for k, v in skim.items() if k == "hem_veto"} == hem, (analysis, name)
        if name in ("et", "mt") and not analysis.startswith("tau_id"):
            assert cuts["lepton_eta"] == skim["lepton_eta"] == "(abs(eta_1) < 2.1)", (analysis, name)  # E31
        if analysis.startswith("tau_id"):
            assert cuts["muon_eta"] == skim["muon_eta"] == "(abs(eta_1) < 2.1)", name  # E2, E31
    frame = pd.DataFrame({"eta_1": [-2.11, -2.09, 2.09, 2.1]})
    assert mask(frame, [tau(run2, era).channel("mt").cuts["lepton_eta"]]).tolist() == [False, True, True, False]


@pytest.mark.parametrize("era", ERAS)
def test_prefiring_weight_and_its_lnn(run2, era):
    """E11, C11: prefiring_wgt on the MC of every era, not on data or embedding; CMS_prefiring from its Up/Dn columns,
    written as lnN (one nuisance for 2016-2018)."""
    analysis = tau(run2, era)
    assert analysis.lnn_variations == ("CMS_prefiring",) == controls(run2, era).lnn_variations
    for name, channel in analysis.channels.items():
        for process in channel.processes:
            kind = channel.kind_of(process)
            assert (process.selection.weights.get("prefiring") == "prefiring_wgt") == (kind == "mc"), (name, process.name)
        variations = [v for v in channel.variations if v.name.startswith("CMS_prefiring")]
        assert variations == [WeightVariation("CMS_prefiringUp", {"prefiring": "L1PreFiringWeight_Up"}), WeightVariation("CMS_prefiringDown", {"prefiring": "L1PreFiringWeight_Dn"})]
    weights = tau_id(run2, era).channel("mt").process("ZL").selection.weights
    assert weights["prefiring"] == "prefiring_wgt" and "prefiring" not in tau_id(run2, era).channel("mt").process("EMB").selection.weights


@pytest.mark.parametrize("era", ERAS)
def test_btag_nuisances_follow_the_four_component_scheme(run2, era):
    """6.5: b/c and light, correlated (no era) and uncorrelated (per era, the 2016 halves separate)."""
    expected = {
        "CMS_btag_fixedWP_bc_correlatedUp": "btag_weight_upart_up_bc_correlated",
        "CMS_btag_fixedWP_light_correlatedDown": "btag_weight_upart_down_light_correlated",
        f"CMS_btag_fixedWP_bc_uncorrelated_{era}Down": "btag_weight_upart_down_bc_uncorrelated",
        f"CMS_btag_fixedWP_light_uncorrelated_{era}Up": "btag_weight_upart_up_light_uncorrelated",
    }
    variations = {v.name: v.replace_weights["btag"] for v in btag_variations(era)}
    assert len(variations) == 8 and expected.items() <= variations.items()
    for analysis in (tau(run2, era), controls(run2, era)):
        for channel in analysis.channels.values():
            assert {v.name for v in channel.variations if isinstance(v, WeightVariation) and "btag" in v.replace_weights} == set(variations)


@pytest.mark.parametrize("era", ERAS)
def test_lumi_follows_the_cholesky_scheme(run2, era):
    """6.6: the LUM POG names and values of the year, MC only (not the embedded sample, not the jet fakes)."""
    analysis = tau(run2, era)
    lumi = {entry.name: entry for entry in analysis.lnn if entry.name.startswith("lumi_")}
    assert {name: entry.value for name, entry in lumi.items()} == LUMI[YEAR[era]]
    for entry in lumi.values():
        assert entry.channels is None and {"ZL", "TTL", SIGNAL} <= set(entry.processes) and not {"EMB", "jetFakes"} & set(entry.processes)


@pytest.mark.parametrize("era", ERAS)
def test_normalisation_uncertainties_per_era(run2, era):
    """6.7: trigger E (MC and embedding separately), tau E, fake factors J; E27 on the embedded lepton ID."""
    year, analysis = YEAR[era], tau(run2, era)
    lnn = {(e.name, e.channels, e.value): set(e.processes) for e in analysis.lnn}
    names = {name for name, *_ in lnn}
    assert {f"CMS_eff_m_trigger_{era}", f"CMS_eff_m_trigger_emb_{era}", f"CMS_eff_e_trigger_{era}", f"CMS_eff_e_trigger_emb_{era}"} <= names
    vs_lepton = "2018" if year == "2016" else era  # K6: VSe and VSmu of 2016 are copies of 2018 (E12)
    assert {f"eff_t_vsLep_$CHANNEL_{vs_lepton}", f"eff_t_vsJet_$CHANNEL_{era}", f"jetFakesNorm_$CHANNEL_{year}"} <= names
    assert lnn[(f"CMS_eff_m_trigger_{era}", ("mt", "em"), 1.02)] >= {"ZL", "TTL", "TT"} and lnn[(f"CMS_eff_m_trigger_emb_{era}", ("mt",), 1.02)] == {"EMB"}
    # E27: the embedded sample shares 0.5 of the MC lepton-ID nuisance and has its own 0.866 part
    assert lnn[("eff_m", ("mt",), 1.01)] == lnn[("eff_m_emb", ("mt",), 1.017)] == lnn[("eff_e", ("et",), 1.01)] == lnn[("eff_e_emb", ("et",), 1.017)] == {"EMB"}
    assert "EMB" not in lnn[("eff_m", ("mt", "em"), 1.02)] | lnn[("eff_e", ("et", "em"), 1.02)]
    # E26: the HH cross-section uncertainties; VH -> tautau carries the VH theory
    assert lnn[("QCDscale_HH", None, (0.7888, 1.0725))] == lnn[("PDF_alphas_HH", None, 1.0247)] == {SIGNAL} and not [n for n in names if n.startswith("mtop")]
    assert VH_TAUTAU in lnn[("QCDscale_VH", None, (0.970, 1.038))] & lnn[("pdf_VH", None, 1.017)]
    for name, channel in analysis.channels.items():
        if name == "em":
            continue
        ff = [v for v in channel.variations if isinstance(v, ColumnVariation) and v.suffix.startswith("__") and "ff" in v.name]
        assert ff and all(v.name.removesuffix("Up").removesuffix("Down").endswith(f"_{name}_{year}") for v in ff), name
        assert TemplateShift(f"CMS_htt_emb_ttbar_{era}", "EMB", "TTT", 0.1) in channel.estimators


def test_crown_shifts_switch_keeps_the_nominal_tree_variations(run2):
    """E34: a nominal-only production has no embedding tau shifts; the FF friend shifts and weight variations stay."""
    nominal, shifted = (tau(run2, "2017", crown_shifts=value).channel("mt") for value in (False, True))
    emb = [v for v in shifted.variations if isinstance(v, ColumnVariation) and "emb" in v.name]
    assert emb and all(v.suffix.endswith(("_2017Up", "_2017Down")) for v in emb)
    assert {v.name for v in shifted.variations} - {v.name for v in nominal.variations} == {v.name for v in emb}
    assert any("CMS_ff_" in v.name for v in nominal.variations) and any("CMS_prefiring" in v.name for v in nominal.variations)


def test_the_correlation_scheme():
    """6.7 with K4 and K6: K no token, E per era (copied 2018 payloads take 2018), J per year."""
    tokens = {family: tuple(era_token(family, era) for era in ERAS) for family in FAMILIES}
    assert tokens["prefiring"] == tokens["pileup"] == tokens["lepton_id"] == tokens["electron_reco"] == tokens["electron_es"] == (None,) * 4
    assert tokens["trigger"] == tokens["embedding"] == tokens["btag_uncorrelated"] == tokens["muon_id_stat"] == ERAS
    assert tokens["fake_factors"] == tokens["recoil"] == ("2016", "2016", "2017", "2018")
    assert tokens["tau_vse"] == tokens["tau_e_es"] == ("2018", "2018", "2017", "2018")  # K6: copies of 2018 (E12)
    assert tokens["tau_vsmu"] == ("2018", "2018", "2018", "2018")
    assert tokens["tau_vslep"] == ("2018", "2018", "2017", "2018")  # the flat VSe + VSmu lnN: a copy where both are
    assert tokens["jes_year"] == ("2016APV", "2016", "2017", "2018")
    assert nuisance("CMS_btag_fixedWP_bc_correlated", "btag_correlated", "2017") == "CMS_btag_fixedWP_bc_correlated"
    assert nuisance("CMS_ff_QCDStatShift_mt", "fake_factors", "2016preVFP") == nuisance("CMS_ff_QCDStatShift_mt", "fake_factors", "2016postVFP") == "CMS_ff_QCDStatShift_mt_2016"
    with pytest.raises(ValueError, match="unknown era"):
        era_token("trigger", "2022")


def test_datacards_carry_the_v7_normalisation_nuisances(run2, tmp_path):
    """The cards of 2016postVFP: Cholesky lumi, prefiring as lnN of the yield ratios, the borrowed VH -> tautau."""
    analysis = tau(run2, "2016postVFP", fit_categories="control")
    channel = analysis.channel("mt")
    hset = HistogramSet()

    def put(process, variation, value):
        hset[HistKey("mt", "nominal", process, "nominal", variation, "yield")] = Histogram([0.5, 1.5], [value], [0.01])

    put("data", "Nominal", 100.0)
    for process in (analysis.signal, *channel.backgrounds()):
        put(process, "Nominal", 10.0)
        if process not in ("EMB", "jetFakes"):
            put(process, "CMS_prefiringUp", 10.2)
            put(process, "CMS_prefiringDown", 9.8)
    card = write_datacard(hset, analysis, ["mt"], tmp_path / "mt", min_background=None).read_text()
    lines = {line.split()[0]: line.split()[1:] for line in card.splitlines() if " lnN " in line}
    processes = card.splitlines()[[i for i, line in enumerate(card.splitlines()) if line.startswith("process")][0]].split()[1:]
    column = dict(zip(processes, range(len(processes))))
    assert lines["lumi_13TeV_1516_l"][1 + column["ZL"]] == "1.0118" and lines["lumi_13TeV_1516_l"][1 + column["EMB"]] == "-"
    assert lines["CMS_prefiring"][1 + column["ZL"]] == "0.9800/1.0200" and lines["CMS_prefiring"][1 + column["jetFakes"]] == "-"
    assert lines["CMS_htt_borrowed_VH_htt_2016postVFP"][1 + column[VH_TAUTAU]] == "1.05"
    assert lines["eff_m"][1 + column["ZL"]] == "1.02" and lines["eff_m"][1 + column["EMB"]] == "1.01"
    assert "CMS_prefiring shape" not in card


def test_vh_tautau_of_2016postVFP_is_borrowed_from_2016preVFP(run2):
    """3.3: the 2016preVFP n-tuples, normalised with the 2016postVFP luminosity, plus an lnN 1.05."""
    analysis = tau(run2, "2016postVFP")
    vh = analysis.channel("mt").samples_of(VH_TAUTAU)
    assert sorted(s.nick for s in vh) == sorted(nick(prefix, "2016preVFP") for prefix in VH_PRIMARIES)
    assert all(s.era == "2016preVFP" for s in vh) and analysis.lumi_pb == LUMI_PB["2016postVFP"]
    assert LnN("CMS_htt_borrowed_VH_htt_2016postVFP", (VH_TAUTAU,), 1.05) in analysis.lnn
    for era in ("2016preVFP", "2017", "2018"):
        other = tau(run2, era)
        assert all(s.era is None for s in other.channel("mt").samples) and len(other.channel("mt").samples_of(VH_TAUTAU)) == 3
        assert not [e for e in other.lnn if "borrowed" in e.name]


def test_borrowing_is_checked(run2, tmp_path):
    assert parse_entry("2016preVFP/ZHToTauTau_X") == ("ZHToTauTau_X", "2016preVFP") and parse_entry("ZHToTauTau_X") == ("ZHToTauTau_X", None)
    with pytest.raises(AnalysisError, match="the 2017 run does not borrow VH_htt from 2016preVFP"):
        samples(run2, [list_name("borrowed", "2016postVFP")], ["mt"], "2017")
    with pytest.raises(AnalysisError, match="a 2017 sample in a 2018 run"):
        samples(run2, [list_name("binned", "2017")], ["mt"], "2018")
    wrong = samples_module.INVENTORY_DIR / "run2_wrong_borrowing.txt"
    wrong.write_text(f"2016preVFP/{nick(VH_PRIMARIES[0], '2017')}\n2015/{nick(VH_PRIMARIES[1], '2016preVFP')}\n")
    with pytest.raises(AnalysisError, match=r"the database has it as a 2017 sample\n.*the 2016postVFP run does not borrow VH_htt from 2015"):
        samples(run2, ["run2_wrong_borrowing"], ["mt"], "2016postVFP")


def test_the_signal_and_the_new_samples_are_routed(run2):
    """6.3: powheg kappa_lambda = 1 is the signal, node_SM its auxiliary cross-check, the other kappa_lambda points have
    no process; SingleElectron data, VH -> tautau, ttW, WZTo4Q and ZZTo2Nu2Q_5f."""
    assert route(nick(POWHEG["1p00"], "2017")) == (SIGNAL, None) and route(f"{POWHEG['0p00']}_RunIISummer20UL17NanoAODv15") == ("HH2B2Tau_kl", None)
    assert route("GluGluToHHTo2B2Tau_TuneCP5_PSWeights_node_SM_13TeV-madgraph-pythia8_RunIISummer20UL17") == (NODE_SM, None)
    assert route("SingleElectron_Run2016F-HIPM_UL2016_NanoAODv15") == ("data", ("et", "ee"))
    for primary in VH_PRIMARIES:
        assert route(primary) == (VH_TAUTAU, None)
    assert route("WplusH_HToBB_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8") == ("VH", None) and route("ZH_HToBB_ZToLL_M-125") == ("VH", None)
    for listed, group in (("TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8", "TTV"), ("TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8", "TTV"),
                        ("WZTo4Q_4f_TuneCP5_13TeV_amcatnloFXFX_pythia8", "VV"), ("ZZTo2Nu2Q_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8", "VV")):
        assert route(listed) == (group, None)
    mt = tau(run2, "2017").channel("mt")
    assert [s.nick for s in mt.samples_of(SIGNAL)] == [nick(POWHEG["1p00"], "2017")]
    assert mt.process(NODE_SM).role == "auxiliary" and len(mt.samples_of(NODE_SM)) == 1
    assert not mt.samples_of("HH2B2Tau_kl") and NODE_SM not in tau(run2, "2017", ml_labels="v7b5", ml_variables=["kinematics"]).ml.processes
    et = tau(run2, "2017").channel("et")
    assert all(s.nick.startswith("SingleElectron_") for s in et.samples_of("data"))


def test_the_node_sm_sample_stands_in_for_the_signal_without_powheg():
    """The 2018 productions up to v6 had node_SM as their signal."""
    mt = build(config(("mt",), sample_lists=["sm2018_binned_v6"], database=DATABASE)).channel("mt")
    (signal,) = mt.samples_of(SIGNAL)
    assert "node_SM" in signal.nick and NODE_SM not in {p.name for p in mt.processes} and VH_TAUTAU not in {p.name for p in mt.processes}


@pytest.mark.parametrize("era", ERAS)
def test_the_signal_from_the_node_sm_sample(run2, era):
    """X7, N9: signal_sample node_sm takes the signal of every channel from the madgraph node_SM sample (R3, and its ML
    export with thesis6) and leaves out the powheg sample."""
    analysis = tau(run2, era, jet_fakes="mc", embedding=False, signal_sample="node_sm", ml_labels="thesis6", ml_variables=["kinematics"])
    for channel in analysis.channels.values():
        (signal,) = channel.samples_of(SIGNAL)
        assert "node_SM" in signal.nick and not any("kl-1p00" in s.nick for s in channel.samples), channel.name
        assert NODE_SM not in {p.name for p in channel.processes} and channel.process(SIGNAL).role == "signal"
    assert analysis.ml.class_of[SIGNAL] == "HH2B2Tau" and analysis.ml.exported(analysis.channel("mt"))[0] == SIGNAL
    assert [s.nick for s in tau(run2, era).channel("mt").samples_of(SIGNAL)] == [nick(POWHEG["1p00"], era)]


@pytest.mark.parametrize("era", ERAS)
def test_the_dy_table_of_every_era(run2, era, monkeypatch):
    """E19, spec 6.10: the DY of each era carries the table of its own dilepton run; an era without a table fails."""
    expected = weights_module.dy_njets_btag_weight(era)
    assert tau(run2, era, dy_correction="njets_btag").channel("tt").process("ZL").selection.weights["dy_correction"] == expected
    monkeypatch.setattr(weights_module, "DY_NJETS_BTAG", {other: table for other, table in DY_NJETS_BTAG.items() if other != era})
    with pytest.raises(AnalysisError, match=f"no DY table for {era}"):
        tau(run2, era, dy_correction="njets_btag")


@pytest.mark.parametrize("era", ERAS)
def test_the_tau_id_systematics_per_era(run2, era):
    """6.1: the CROWN shifts of the era, the datacard names of MorphingTauID2017 (Run<era>), no HEM (E29), prefiring
    for every era (E11) from CMS_ecal_prefiring_<year>, the electron ES without era (K4), the top-pT shape with
    against without the weight (K9)."""
    tag = f"Run{era}"
    shifts = {name: shift for name, shift, _ in crown_shifts(era)}
    assert shifts["CMS_prefiring"] == f"CMS_ecal_prefiring_{YEAR[era]}" and shifts[f"CMS_eff_m_trigger_{tag}"] == f"CMS_eff_m_trigger_{era}"
    assert shifts[f"CMS_scale_j_Absolute_{tag}"] == f"CMS_scale_j_Absolute_{era}" and shifts["CMS_PileUp"] == f"CMS_pileup_{era}"
    assert not [name for name in shifts if "HEM" in name] and not [shift for shift in shifts.values() if "HEM" in shift]
    assert dict(unused_shifts(era))["CMS_scale_e"] == f"CMS_scale_e_{era}" and dict(unused_shifts(era))["CMS_res_e"] == f"CMS_res_e_{era}"
    assert ("CMS_htt_ttbarShape", "top_pt", "(2.0 * topPtReweightWeight - 1.0)", "1.0") in weight_shifts(era)  # K9: mirror of without
    channel = tau_id(run2, era).channel("mt")
    suffixes = {v.name: v.suffix for v in channel.variations if isinstance(v, ColumnVariation) and v.suffix}
    assert suffixes["CMS_prefiringUp"] == f"__CMS_ecal_prefiring_{YEAR[era]}Up" and suffixes[f"CMS_fake_m_WH1_{tag}Down"] == f"__CMS_fake_t_DeepTau2018v2p5_VSmu_wheel1_{era}Down"
    assert all(suffix.removesuffix("Up").removesuffix("Down").endswith((f"_{era}", f"_{YEAR[era]}")) or suffix.startswith("__CMS_scale_j_") for suffix in suffixes.values())
    assert TemplateShift(f"CMS_emb_ttbar_contamination_{tag}", "EMB", "TTT", 0.1) in channel.estimators
    weights = channel.process("TTJ").selection.weights
    assert weights["prefiring"] == "prefiring_wgt" and weights["trigger"] == TRIGGER[era]["tau_id"].mc_weight


def test_the_mc_weights_of_every_era_read_their_trigger(run2):
    for era in ERAS:
        for channel in TAU_CHANNELS:
            assert columns_of(mc_weights(channel, era).values()) <= branches(channel, era), (era, channel)
            assert columns_of(embedding_weights(channel, era).values()) <= embedding_branches(channel, era), (era, channel)
    assert np.isclose(0.866**2 + 0.5**2, 1.0, atol=1e-3)  # E27: the embedded lepton-ID uncertainty stays 2 %
