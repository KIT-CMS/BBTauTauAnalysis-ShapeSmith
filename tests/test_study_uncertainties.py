"""The switch study_uncertainties: the uncertainties sized in the v7b studies (study_uncertainties.py) in the cards of
the v7b fake-factor runs, and nowhere without the switch."""
from pathlib import Path

import pandas as pd
import pytest
from shapesmith.datacards import write_datacard
from shapesmith.expressions import evaluate, shift
from shapesmith.histogram import HistKey, Histogram, HistogramSet
from shapesmith.model import ColumnVariation

from bbtautau_shapesmith import study_uncertainties as sizes
from bbtautau_shapesmith.constants import ERAS, LT_CHANNELS, RECOIL_ERAS, YEAR
from bbtautau_shapesmith.processes import EMBEDDED, SIGNAL
from tests.test_run2 import run2, tau  # noqa: F401 (the fixture)

CROWN_PAYLOAD = Path("/work/sdaigler/bbtautau/KingMaker/CROWN/analysis_configurations/bbtautau/payloads/tau_embedding/Run2-{era}-UL-NanoAODv15/DeepTau2018v2p5_id_es_embedding{era}UL.json.gz")
STUDY_FAMILIES = ("CMS_scale_t_", "CMS_eff_t_vsJet", "CMS_eff_t_trigger_ditau", "CMS_eff_m_stat", "CMS_eff_e_reco", "CMS_htt_met_mt_acceptance", "CMS_htt_hipm", "CMS_ff_2016half")


def v7b(database, era, **switches):
    """The v7b fake-factor run: embedding in et and mt, tt from MC (E33), nominal only, with the study uncertainties."""
    return tau(database, era, embedding=["et", "mt"], **{"study_uncertainties": True, **switches})


def lnn_of(analysis, name):
    return [e for e in analysis.lnn if e.name == name]


def derived(analysis, channel, name):
    return next(v for v in analysis.channel(channel).variations if v.name == name)


@pytest.mark.parametrize("era", ERAS)
def test_without_the_switch_nothing_changes(run2, era):
    analysis = tau(run2, era, embedding=["et", "mt"])
    assert not [e for e in analysis.lnn if e.name.startswith(STUDY_FAMILIES)]
    assert lnn_of(analysis, f"eff_t_vsJet_$CHANNEL_{era}")  # the flat MC vsJet lnN stays
    assert not [v for channel in analysis.channels.values() for v in channel.variations if isinstance(v, ColumnVariation) and v.derived]


@pytest.mark.parametrize("era", ERAS)
def test_embedding_tau_id_is_an_exact_weight_factor_per_decay_mode_category(run2, era):
    analysis = v7b(run2, era)
    frame = pd.DataFrame({"id_wgt_tau_vsJet_Medium_2": [0.9, 0.9, 0.9, 0.9], "tau_decaymode_2": [0, 1, 10, 11]})
    for channel, wp in (("et", "Tight"), ("mt", "VVLoose")):
        for token, hit in (("0", [1, 0, 0, 0]), ("1", [0, 1, 0, 0]), ("1011", [0, 0, 1, 1])):
            down, up = sizes.EMBEDDING_TAU_ID[era][wp][token]
            for direction, ratio in (("Up", up), ("Down", down)):
                variation = derived(analysis, channel, f"CMS_eff_t_emb_dm{token}_vsEle{wp}_{era}{direction}")
                assert variation.applies_to == ("embedding",)
                weight = evaluate(frame, shift("id_wgt_tau_vsJet_Medium_2", variation, set()))
                assert list(weight) == pytest.approx([0.9 * (ratio if h else 1.0) for h in hit])
    assert not [v for v in analysis.channel("tt").variations if v.name.startswith("CMS_eff_t_emb")]  # tt: MC genuine taus


@pytest.mark.parametrize("era", ERAS)
@pytest.mark.parametrize("wjets", [False, True])
def test_ff_decay_mode_nonclosure_shifts_the_fake_factor_of_each_leg(run2, era, wjets):
    analysis = v7b(run2, era, ff_processes=["QCD", "Wjets", "ttbar"] if wjets else ["QCD", "ttbar"])
    table = sizes.FF_DM_NONCLOSURE["QCD_Wjets_ttbar" if wjets else "QCD_ttbar"][YEAR[era]]
    frame = pd.DataFrame({"fake_factor": [0.1] * 5, "fake_factor_1": [0.2] * 5, "fake_factor_2": [0.3] * 5,
                          "tau_decaymode_1": [0, 1, 2, 10, 11], "tau_decaymode_2": [11, 10, 2, 1, 0]})
    bins = {0: 0, 1: 1, 2: 1, 10: 10, 11: 11}
    for direction, sign in (("Up", 1), ("Down", -1)):
        mt = derived(analysis, "mt", f"CMS_ff_dm_nonclosure_mt_{YEAR[era]}{direction}")
        assert mt.applies_to == ("data", "mc", "embedding")
        expected = [0.1 * (1 + sign * table["mt"]["2"][bins[dm]]) for dm in frame.tau_decaymode_2]
        assert list(evaluate(frame, shift("fake_factor", mt, set()))) == pytest.approx(expected)
        tt = derived(analysis, "tt", f"CMS_ff_dm_nonclosure_tt_{YEAR[era]}{direction}")
        for leg, nominal in ((1, 0.2), (2, 0.3)):
            expected = [nominal * (1 + sign * table["tt"][str(leg)][bins[dm]]) for dm in frame[f"tau_decaymode_{leg}"]]
            assert list(evaluate(frame, shift(f"fake_factor_{leg}", tt, set()))) == pytest.approx(expected)
    assert not [v for v in analysis.channel("et").variations if "dm_nonclosure" in v.name]  # et closes


@pytest.mark.parametrize("era", ERAS)
def test_the_lnn_follow_the_correlation_scheme(run2, era):
    analysis = v7b(run2, era)
    names = {e.name for e in analysis.lnn}
    assert not [name for name in names if name.startswith("eff_t_vsJet_")]  # replaced by the POG components
    assert "CMS_eff_t_vsJet_syst_alleras" in names and "CMS_eff_e_reco" in names  # K
    assert {f"CMS_eff_t_vsJet_{era}", f"CMS_eff_m_stat_{era}", f"CMS_scale_t_dm1_{era}", f"CMS_eff_t_vsJet_syst_TES_dm1_{era}"} <= names  # E
    trigger = lnn_of(analysis, f"CMS_eff_t_trigger_ditau_dm1_{era}")
    assert trigger and all(e.channels == ("tt",) for e in trigger)
    for token in ("0", "1", "1011"):
        for channel, wp in (("et", "Tight"), ("mt", "VVLoose")):
            (es,) = lnn_of(analysis, f"CMS_scale_t_emb_dm{token}_vsEle{wp}_{era}")
            assert es.processes == (EMBEDDED,) and es.channels == (channel,) and es.value == sizes.EMBEDDING_TAU_ES[era][channel][token]
    met = lnn_of(analysis, f"CMS_htt_met_mt_acceptance_{era}")
    if era in RECOIL_ERAS:
        assert not met
    else:
        (met,) = met
        # DY and W of the run; in et and mt of the fake-factor run that is ZL
        assert "ZL" in met.processes and EMBEDDED not in met.processes and met.channels == LT_CHANNELS and met.value == 1.03
    hipm = [e for e in analysis.lnn if "hipm" in e.name]
    if era == "2016preVFP":
        assert {e.name: e.channels for e in hipm} == {"CMS_htt_hipm_eff_m_2016preVFP": ("mt", "em"), "CMS_htt_hipm_eff_e_2016preVFP": ("et", "em")}
        assert all(SIGNAL in e.processes and "ZL" not in e.processes and "DY" not in e.processes and EMBEDDED not in e.processes for e in hipm)
    else:
        assert not hipm
    halves = {e.name: e for e in analysis.lnn if "2016half" in e.name}
    if era.startswith("2016"):
        assert set(halves) == {f"CMS_ff_2016half_{channel}_{era}" for channel in ("et", "mt", "tt")}
        assert all(e.processes == ("jetFakes",) and e.value == sizes.FF_HALVES["QCD_ttbar"][era][e.channels[0]] for e in halves.values())
    else:
        assert not halves


def test_the_tables_cover_every_era_channel_and_class():
    for era in ERAS:
        for channel, classes in sizes.MC_TAU_CLASSES.items():
            assert set(sizes.MC_TAU[era][channel]) == set(classes)
            for table in sizes.MC_TAU[era][channel].values():
                values = [v for value in table.values() for v in (value if isinstance(value, tuple) else (value,))]
                assert all(0.85 < v < 1.15 for v in values)
                assert ("trigger_dm0" in table) == (channel == "tt")
        assert set(sizes.EMBEDDING_TAU_ES[era]) == set(LT_CHANNELS) and set(sizes.LEPTONS[era]) == {"et", "mt", "em"}
    assert set(sizes.FF_HALVES["QCD_ttbar"]) == {"2016preVFP", "2016postVFP"}


@pytest.mark.parametrize("era", ERAS)
def test_embedding_tau_id_equals_the_crown_payload(era):
    path = Path(str(CROWN_PAYLOAD).format(era=era))
    correctionlib = pytest.importorskip("correctionlib")
    if not path.exists():
        pytest.skip("CROWN checkout not available")
    vsjet = correctionlib.CorrectionSet.from_file(str(path))["DeepTau2018v2p5VSjet"]
    for wp, tokens in sizes.EMBEDDING_TAU_ID[era].items():
        for token, (down, up) in tokens.items():
            for dm in {"0": (0,), "1": (1,), "1011": (10, 11)}[token]:
                for pt in (25.0, 60.0, 150.0):  # flat in pT
                    nominal = vsjet.evaluate(pt, dm, 5, "Medium", wp, "nom", "dm")
                    assert vsjet.evaluate(pt, dm, 5, "Medium", wp, "up", "dm") / nominal == pytest.approx(up, abs=1e-4)
                    assert vsjet.evaluate(pt, dm, 5, "Medium", wp, "down", "dm") / nominal == pytest.approx(down, abs=1e-4)


def test_the_datacard_of_2016preVFP(run2, tmp_path):
    """Dry run of the cards of the v7b fake-factor run (one-bin categories): the study lnN per process column, the
    embedding tau ID and the fake-factor decay-mode non-closure as shapes."""
    analysis = v7b(run2, "2016preVFP", fit_categories="control")
    hset = HistogramSet()
    for channel in ("mt", "tt"):
        def put(process, variation, value):
            hset[HistKey(channel, "nominal", process, "nominal", variation, "yield")] = Histogram([0.5, 1.5], [value], [0.01])

        put("data", "Nominal", 100.0)
        for process in (analysis.signal, *analysis.channel(channel).backgrounds()):
            put(process, "Nominal", 10.0)
        for direction, value in (("Up", 10.3), ("Down", 9.7)):
            put("jetFakes", f"CMS_ff_dm_nonclosure_{channel}_2016{direction}", value)
            if channel == "mt":
                put(EMBEDDED, f"CMS_eff_t_emb_dm1_vsEleVVLoose_2016preVFP{direction}", value)
    card = write_datacard(hset, analysis, ["mt", "tt"], tmp_path / "cmb", min_background=None).read_text()
    lines = card.splitlines()
    bins = lines[[i for i, line in enumerate(lines) if line.startswith("bin ")][1]].split()[1:]
    processes = lines[[i for i, line in enumerate(lines) if line.startswith("process")][0]].split()[1:]
    column = {(b, p): i for i, (b, p) in enumerate(zip(bins, processes))}
    rows = {line.split()[0]: line.split()[2:] for line in lines if " lnN " in line or " shape " in line}
    mt, tt = "htt_mt_1_2016preVFP", "htt_tt_1_2016preVFP"
    assert rows["CMS_htt_hipm_eff_m_2016preVFP"][column[mt, "TTL"]] == "1.011" and rows["CMS_htt_hipm_eff_m_2016preVFP"][column[mt, "ZL"]] == "-"
    assert rows["CMS_htt_met_mt_acceptance_2016preVFP"][column[mt, "ZL"]] == "1.03" and rows["CMS_htt_met_mt_acceptance_2016preVFP"][column[tt, "ZL"]] == "-"
    assert rows["CMS_ff_2016half_tt_2016preVFP"][column[tt, "jetFakes"]] == f"{sizes.FF_HALVES['QCD_ttbar']['2016preVFP']['tt']:g}"
    down, up = sizes.MC_TAU["2016preVFP"]["tt"]["genuine"]["trigger_dm1"]
    assert rows["CMS_eff_t_trigger_ditau_dm1_2016preVFP"][column[tt, "TTT"]] == f"{down:g}/{up:g}"
    assert rows["CMS_eff_t_trigger_ditau_dm1_2016preVFP"][column[mt, "TTL"]] == "-"
    assert rows["CMS_eff_t_vsJet_syst_alleras"][column[tt, SIGNAL]] != "-" and rows["CMS_eff_t_vsJet_syst_alleras"][column[mt, EMBEDDED]] == "-"
    assert rows["CMS_scale_t_emb_dm1_vsEleVVLoose_2016preVFP"][column[mt, EMBEDDED]] != "-"
    assert rows["CMS_eff_t_emb_dm1_vsEleVVLoose_2016preVFP"][column[mt, EMBEDDED]] == "1"
    assert rows["CMS_ff_dm_nonclosure_tt_2016"][column[tt, "jetFakes"]] == "1"
    assert "eff_t_vsJet_tt_2016preVFP" not in rows
