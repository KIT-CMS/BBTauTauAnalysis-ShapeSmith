"""The datacards of the v7b fake-factor runs after the first fits (2026-10-19): the fake-factor friend shifts
symmetrised where one-sided or strongly asymmetric (switch symmetrise_ff_shifts), the no-op nuisances left out, and the
DY-table nuisance of spec 6.7 (family E) wherever the DY table is applied."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from shapesmith.datacards import write_datacard
from shapesmith.histogram import INCLUSIVE, NOMINAL_VARIATION, HistKey, Histogram, HistogramSet
from shapesmith.model import AnalysisError

from bbtautau_shapesmith import dilepton, samples as samples_module
from bbtautau_shapesmith.constants import DILEPTON_CHANNELS, DY_NJETS_BTAG, DY_TABLE_LNN, ERAS, FF_SHIFTS_LT, FF_SHIFTS_TT, FF_SHIFTS_W, YEAR
from bbtautau_shapesmith.dy_correction import TableUncertainty, table_uncertainty
from bbtautau_shapesmith.processes import EMBEDDED
from tests.helpers import REPO, config
from tests.run2 import install, list_name
from tests.test_run2 import run2, tau  # noqa: F401 (the fixture)

V7B_FF_RUNS = ("smrun2_binned_v7b_ff", "smrun2_binned_v7b_ff_w")


def v7b(database, era, **switches):
    return tau(database, era, embedding=["et", "mt"], **{"study_uncertainties": True, "dy_correction": "njets_btag", "fit_categories": "control", **switches})


@pytest.mark.parametrize("era", ERAS)
@pytest.mark.parametrize("wjets", [False, True])
def test_the_switch_lists_the_fake_factor_friend_shifts(run2, era, wjets):
    processes = ["QCD", "Wjets", "ttbar"] if wjets else ["QCD", "ttbar"]
    analysis = v7b(run2, era, symmetrise_ff_shifts=True, ff_processes=processes)
    year = YEAR[era]
    expected = {f"CMS_ff_{key}_{channel}_{year}" for channel in ("et", "mt") for key in FF_SHIFTS_LT + (FF_SHIFTS_W if wjets else ())}
    expected |= {f"CMS_ff_{key}_tt_{year}" for key in FF_SHIFTS_TT}
    assert set(analysis.symmetrised_variations) == expected and analysis.asymmetry_ratio == 2.0
    assert f"CMS_ff_QCD_non_closure_CorrSystMCShift_mt_{year}" in analysis.symmetrised_variations
    assert not [name for name in analysis.symmetrised_variations if "dm_nonclosure" in name or "2016half" in name]  # only the friend shifts
    varied = {v.name for channel in analysis.channels.values() for v in channel.variations}
    assert all({f"{name}Up", f"{name}Down"} <= varied for name in analysis.symmetrised_variations)
    assert v7b(run2, era, ff_processes=processes).symmetrised_variations == ()


def test_the_switch_needs_fake_factors(run2):
    with pytest.raises(Exception, match="symmetrise_ff_shifts requires jet_fakes: ff"):
        tau(run2, "2018", jet_fakes="mc", embedding=["et", "mt"], symmetrise_ff_shifts=True)


def test_only_the_v7b_fake_factor_runs_symmetrise():
    on = sorted(path.stem for path in (REPO / "configs").glob("*.yaml") if (yaml.safe_load(path.read_text()) or {}).get("switches", {}).get("symmetrise_ff_shifts"))
    assert on == sorted(V7B_FF_RUNS + ("smrun2_binned_v7b_ff_nn",))  # and the copy of the FF run with the NN export


@pytest.mark.parametrize("era", ERAS)
def test_the_dy_table_nuisance_sits_on_every_corrected_dy_process(run2, era):
    analysis = v7b(run2, era)
    (entry,) = [e for e in analysis.lnn if e.name.startswith("CMS_htt_dy_table")]
    assert entry.name == f"CMS_htt_dy_table_{era}" and entry.value == DY_TABLE_LNN[era]  # family E: the 2016 halves separate
    assert set(entry.processes) == {"ZTT", "ZL", "DY"} and EMBEDDED not in entry.processes  # ZTT of tt (MC), ZL, em DY; the FF runs have no ZJ
    assert not [e for e in v7b(run2, era, dy_correction="none").lnn if e.name.startswith("CMS_htt_dy_table")]
    with_mc_fakes = tau(run2, era, jet_fakes="mc", embedding=["et", "mt"], dy_correction="njets_btag")
    (entry,) = [e for e in with_mc_fakes.lnn if e.name.startswith("CMS_htt_dy_table")]
    assert {"ZL", "ZJ", "ZTT", "DY"} <= set(entry.processes)


def test_every_era_with_a_dy_table_has_its_nuisance():
    assert set(DY_TABLE_LNN) == set(DY_NJETS_BTAG) == set(ERAS)
    assert all(1.0 < value < 1.05 for value in DY_TABLE_LNN.values())


def test_an_era_without_the_dy_table_nuisance_fails(run2, monkeypatch):
    from bbtautau_shapesmith import systematics

    monkeypatch.setattr(systematics, "DY_TABLE_LNN", {})
    with pytest.raises(AnalysisError, match="no DY-table uncertainty for 2018"):
        v7b(run2, "2018")


def test_the_cards_of_2018(run2, tmp_path):
    """Dry run of the one-bin cards: the mt QCD non-closure shift of 2018 (-7.8/+27.9 %) symmetrised, a mild shift and
    the non-fake-factor shifts untouched, the ttbar MC shift (equal to the nominal) left out, the DY-table lnN."""
    analysis = v7b(run2, "2018", symmetrise_ff_shifts=True)
    hset = HistogramSet()
    for channel in ("mt", "tt", "em"):
        category = "em_tt" if channel == "em" else "nominal"

        def put(process, variation, value):
            hset[HistKey(channel, category, process, "nominal", variation, "yield")] = Histogram([0.5, 1.5], [value], [0.01])

        put("data", "Nominal", 100.0)
        for process in (analysis.signal, *analysis.channel(channel).backgrounds()):
            put(process, "Nominal", 10.0)
        if channel == "mt":
            for direction, value in (("Up", 9.216), ("Down", 12.787)):
                put("jetFakes", f"CMS_ff_QCD_non_closure_CorrSystMCShift_mt_2018{direction}", value)
            for direction, value in (("Up", 10.5), ("Down", 9.6)):
                put("jetFakes", f"CMS_ff_QCDStatShift_mt_2018{direction}", value)
                put("jetFakes", f"CMS_ff_dm_nonclosure_mt_2018{direction}", value + 0.6)  # one-sided, but no friend shift
            for direction in ("Up", "Down"):
                put("jetFakes", f"CMS_ff_ttbarSystMCShift_mt_2018{direction}", 10.0)
    card = write_datacard(hset, analysis, ["mt", "tt", "em"], tmp_path / "all", min_background=None)
    text = card.read_text()
    rows = {line.split()[0]: line.split()[2:] for line in text.splitlines() if " lnN " in line or " shape " in line}
    lines = text.splitlines()
    bins = lines[[i for i, line in enumerate(lines) if line.startswith("bin ")][1]].split()[1:]
    processes = lines[[i for i, line in enumerate(lines) if line.startswith("process")][0]].split()[1:]
    column = {(b, p): i for i, (b, p) in enumerate(zip(bins, processes))}
    dy = rows["CMS_htt_dy_table_2018"]
    assert dy[column["htt_mt_1_2018", "ZL"]] == dy[column["htt_tt_1_2018", "ZTT"]] == dy[column["htt_em_1_2018", "DY"]] == f"{DY_TABLE_LNN['2018']:g}"
    assert dy[column["htt_mt_1_2018", EMBEDDED]] == dy[column["htt_tt_1_2018", "TTT"]] == "-"
    assert "CMS_ff_ttbarSystMCShift_mt_2018" not in rows and "CMS_ff_QCD_non_closure_CorrSystMCShift_mt_2018" in rows
    changes = json.loads((card.parent / "template_changes.json").read_text())
    assert changes["no_ops"] == ["CMS_ff_ttbarSystMCShift_mt_2018"]
    (record,) = changes["symmetrised"]
    assert record["nuisance"] == "CMS_ff_QCD_non_closure_CorrSystMCShift_mt_2018" and record["after"] == pytest.approx([1.17855, 0.82145])
    import uproot

    with uproot.open(card.parent / "common" / "htt_input_2018.root") as f:
        assert f["htt_mt_1_2018/jetFakes_CMS_ff_QCDStatShift_mt_2018Up"].values().tolist() == [10.5]
        assert f["htt_mt_1_2018/jetFakes_CMS_ff_dm_nonclosure_mt_2018Down"].values().tolist() == [10.2]


EDGES = (1.5, 2.5, 3.5, 4.5, 10.5)  # n_jets 2, 3, 4, 5-10


@pytest.fixture(scope="module")
def dilepton_2017(tmp_path_factory):
    directory = tmp_path_factory.mktemp("dy_uncertainty")
    database = install(directory)
    patch = pytest.MonkeyPatch()
    patch.setattr(samples_module, "INVENTORY_DIR", directory)
    yield dilepton.build(config(DILEPTON_CHANNELS, "bbtautau_shapesmith.dilepton:build", era="2017", database=database, sample_lists=[list_name("binned", "2017")]))
    patch.undo()


def _dilepton(analysis, flavour_factor, dy=100.0):
    """DY `dy` per jet bin and region, ttbar 10 (em: data/MC 1), other backgrounds 1; the data of mm = DY times the
    table x flavour_factor["mm"], of ee likewise (data variances = data)."""
    hset = HistogramSet()
    table = {1: (1.2, 1.2, 1.2), 2: (1.5, 1.5, 1.5)}
    for name in DILEPTON_CHANNELS:
        channel = analysis.channel(name)
        for region, row in (("btag0", 1), ("btag1", 1), ("btag2", 2), ("btag3p", 2)):
            yields = {p: np.full(4, dy if p == "DY" else 10.0 if p == "TT" else 1.0) for p in channel.backgrounds()}
            data = sum(yields.values()) + (0.0 if name == "em" else yields["DY"] * (table[row][0] * flavour_factor[name] - 1))
            for process, values in {**yields, channel.data(): data}.items():
                hset[HistKey(name, INCLUSIVE, process, region, NOMINAL_VARIATION, "n_jets")] = Histogram(EDGES, values, np.where(values > 0, values, 0) if process == channel.data() else np.zeros(4))
    return hset, {0: (1.0, 1.0, 1.0), **table}


def test_the_table_uncertainty_is_its_statistics_and_the_flavour_closure(dilepton_2017):
    """mm 2 % above the table, ee 2 % below: residuals +-2 %; the statistical part from the data Poisson of the cells
    (MC without variance here), weighted with the DY of btag1, btag2 and btag3p."""
    hset, table = _dilepton(dilepton_2017, {"mm": 1.02, "ee": 0.98, "em": 1.0})
    result = table_uncertainty(hset, dilepton_2017, table=table)
    assert result.residuals["mm"][0] == pytest.approx(0.02, abs=1e-9) and result.residuals["ee"][0] == pytest.approx(-0.02, abs=1e-9)
    assert result.residual == pytest.approx(0.02)
    # cell statistics: data of mm + ee per cell, over the DY of the cell (n_jets >= 4: two bins)
    data = {1: 2 * (100 * 1.2 + 10 + sum(1.0 for p in dilepton_2017.channel("mm").backgrounds() if p not in ("DY", "TT"))),
            2: 2 * (100 * 1.5 + 10 + sum(1.0 for p in dilepton_2017.channel("mm").backgrounds() if p not in ("DY", "TT")))}
    weights = {1: np.array([200.0, 200.0, 400.0]), 2: np.array([400.0, 400.0, 800.0])}  # btag1; btag2 + btag3p
    errors = {1: np.sqrt(np.array([1, 1, 2]) * data[1]) / np.array([200.0, 200.0, 400.0]), 2: np.sqrt(np.array([1, 1, 2]) * data[2]) / np.array([200.0, 200.0, 400.0])}
    expected = np.sqrt(sum(np.sum((weights[r] * errors[r]) ** 2) for r in (1, 2))) / sum(np.sum(weights[r] * np.array(table[r])) for r in (1, 2))
    assert result.stat == pytest.approx(expected, rel=1e-6)
    assert result.total == pytest.approx(np.hypot(expected, 0.02)) and result.lnn == round(1 + result.total, 3)
    closing, _ = _dilepton(dilepton_2017, {"mm": 1.0, "ee": 1.0, "em": 1.0})
    assert table_uncertainty(closing, dilepton_2017, table=table).residual == pytest.approx(0.0, abs=1e-12)


def test_the_sizes_in_constants_follow_the_rounding_of_the_script():
    result = TableUncertainty("2017", 0.0029, {"mm": (0.0115, 0.0038), "ee": (-0.0165, 0.0044)})
    assert result.lnn == DY_TABLE_LNN["2017"]
