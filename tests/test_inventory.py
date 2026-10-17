"""The inventories (spec 3.2, 6.11): every list resolves in the fixture database, which holds exactly their nicks with
the entries of the sibling database; the v7 lists of every era are byte-identical copies of the CROWN lists and follow
the list rules of 3.1/3.2; the borrowed lists (3.3)."""
import json
import os
from pathlib import Path

import pytest
from shapesmith.samples import normalisation

from bbtautau_shapesmith.constants import BTAG_COMPONENTS, DILEPTON_CHANNELS, TAU_CHANNELS
from bbtautau_shapesmith.samples import inventory_nicks
from tests.helpers import BRANCHES, DATABASE, EMBEDDING_LIST, ERAS, INVENTORY, REPO, branches, fixture_branches
from tests.run2 import VH_TAUTAU, list_name, nick

INVENTORIES = sorted(INVENTORY.glob("*.txt"))
V7_KINDS = ("binned", "binned_mc", "embedding", "tau_id")
# the KingMaker lists of the CROWN checkout of the workspace (KingMaker/CROWN beside ShapeSmith/) and the sibling
# sample database (README "Expected layout"); the tests that compare with them skip where they are absent (CI)
CROWN_LISTS = Path(os.environ.get("CROWN_SAMPLE_LISTS", REPO.parents[1] / "KingMaker" / "CROWN" / "analysis_configurations" / "bbtautau" / "sample_list"))
SIBLING_DATABASE = REPO.parent / "KingMaker_sample_database" / "nanoAOD_v15" / "datasets.json"
EMBEDDING_COUNTS = {"2016preVFP": 15, "2016postVFP": 9, "2017": 15, "2018": 12}  # 3 final states x run periods (no 2016B ver1)
DATA_STREAMS = {"2016preVFP": ("SingleMuon", "SingleElectron", "Tau"), "2016postVFP": ("SingleMuon", "SingleElectron", "Tau"),
                "2017": ("SingleMuon", "SingleElectron", "Tau"), "2018": ("SingleMuon", "EGamma", "Tau")}
PERIODS = {"2016preVFP": 5, "2016postVFP": 3, "2017": 5, "2018": 4}
DY_BINS = tuple(f"DYJetsToLL_LHEFilterPtZ-{b}_" for b in ("0To50", "50To100", "100To250", "250To400", "400To650", "650ToInf"))
ZZ_2NU2Q = ("ZZTo2Nu2Q_5f_", "ZZTo2Q2Nu_")


def database() -> dict:
    return json.loads(DATABASE.read_text())


def v7_nicks(kind: str, era: str) -> list[str]:
    return list(inventory_nicks(list_name(kind, era)))


@pytest.mark.parametrize("path", INVENTORIES, ids=[p.stem for p in INVENTORIES])
def test_inventory_resolves_by_nick(path):
    nicks = inventory_nicks(path.stem)
    assert nicks or path.stem.endswith("_borrowed")
    assert set(normalisation(DATABASE, nicks)) == set(nicks)


def test_fixture_database_holds_exactly_the_inventory_nicks():
    assert set(database()) == {n for path in INVENTORIES for n in inventory_nicks(path.stem)}


def test_the_v7_inventories():
    """Per era the four KingMaker lists and the borrowed list, nothing else of v7."""
    v7 = {p.stem for p in INVENTORIES if p.stem.startswith("sm_")}
    assert v7 == {list_name(kind, era) for era in ERAS for kind in (*V7_KINDS, "borrowed")}


@pytest.mark.skipif(not SIBLING_DATABASE.exists(), reason=f"no sibling sample database {SIBLING_DATABASE}")
def test_the_fixture_entries_are_those_of_the_sibling_database():
    """The sibling checkout is the database of the v7 runs (KingMaker_sample_database v7-run2-registration 0de63ebe or
    later): every inventory nick is there with the fixture's entry."""
    sibling = json.loads(SIBLING_DATABASE.read_text())
    assert {n: entry for n, entry in database().items()} == {n: sibling.get(n) for n in database()}


@pytest.mark.skipif(not CROWN_LISTS.exists(), reason=f"no CROWN sample lists {CROWN_LISTS}")
@pytest.mark.parametrize("era", ERAS)
@pytest.mark.parametrize("kind", V7_KINDS)
def test_the_v7_inventories_are_copies_of_the_crown_lists(era, kind):
    name = list_name(kind, era)
    assert (INVENTORY / f"{name}.txt").read_bytes() == (CROWN_LISTS / f"{name}.txt").read_bytes()


@pytest.mark.parametrize("era", ERAS)
def test_every_v7_nick_is_a_sample_of_its_era(era):
    entries = database()
    for kind in V7_KINDS:
        assert {str(entries[n]["era"]) for n in v7_nicks(kind, era)} == {era}, kind


@pytest.mark.parametrize("era", ERAS)
def test_the_binned_list_of_an_era(era):
    """3.1/3.2: data of the era's streams; DY in the six pT(Z) bins plus the inclusive M-50 (npartons == 0) and M-10to50;
    W 0J/1J/2J; the powheg signal and node_SM; VH -> tautau except 2016postVFP; one ZZ -> 2nu2q sample; TTWJets for
    ttWJets; no ZH -> bb with Z -> bb (in ZToQQ); the MC list is the binned list without the data."""
    entries, nicks = database(), v7_nicks("binned", era)
    data = [n for n in nicks if entries[n]["sample_type"] == "data"]
    assert sorted({n.split("_Run")[0] for n in data}) == sorted(DATA_STREAMS[era]) and len(data) == len(DATA_STREAMS[era]) * PERIODS[era]
    assert not [n for n in data if "ver1" in n or n == "SingleMuon_Run2016B-HIPM_UL2016_NanoAODv15-v1"]  # 2016B ver1: no golden lumi
    assert sorted(v7_nicks("binned_mc", era)) == sorted(set(nicks) - set(data))
    for prefixes, count in ((DY_BINS, 1), (("DYJetsToLL_M-50_", "DYJetsToLL_M-10to50_"), 1), (("WJetsToLNu_0J_", "WJetsToLNu_1J_", "WJetsToLNu_2J_"), 1),
                            (("GluGluToHHTo2B2T_kl-1p00_", "GluGluToHHTo2B2Tau_TuneCP5_PSWeights_node_SM_"), 1), (("TTWJetsToLNu_", "TTWJetsToQQ_", "WZTo4Q_"), 1)):
        for prefix in prefixes:
            assert sum(n.startswith(prefix) for n in nicks) == count, prefix
    assert not [n for n in nicks if n.startswith(("WJetsToLNu_TuneCP5", "ttWJets", "ttZJets", "TTZToNuNu", "ZH_HToBB_ZToBB", "ggZH_HToBB_ZToBB", "GluGluToHHTo2B2T_kl-0p00"))]
    assert sum(n.startswith(ZZ_2NU2Q) for n in nicks) == 1 and any(n.startswith(ZZ_2NU2Q[era == "2018"]) for n in nicks)
    vh = [n for n in nicks if n.startswith(VH_TAUTAU)]
    assert len(vh) == (0 if era == "2016postVFP" else 3) and {entries[n]["sample_type"] for n in vh} <= {"rem_htautau"}


@pytest.mark.parametrize("era", ERAS)
def test_the_embedding_list_of_an_era(era):
    """Every embedded tau final state (eltau, mutau, tautau) of every run period of the era, sorted."""
    nicks = v7_nicks("embedding", era)
    assert nicks == sorted(nicks) and len(nicks) == EMBEDDING_COUNTS[era] and not [n for n in nicks if "ver1" in n]
    periods = {}
    for n in nicks:
        state = next(token for token in ("_eltau_", "_mutau_", "_tautau_") if token in n)
        periods.setdefault(n.split("_cwinter")[0], set()).add(state)  # EmbeddingRun2016-HIPM_B_ver2, EmbeddingRun2018C
    assert len(periods) == EMBEDDING_COUNTS[era] // 3 and all(states == {"_eltau_", "_mutau_", "_tautau_"} for states in periods.values())


@pytest.mark.skipif(not SIBLING_DATABASE.exists(), reason=f"no sibling sample database {SIBLING_DATABASE}")
@pytest.mark.parametrize("era", ERAS)
def test_the_embedding_list_follows_the_sample_list_rule(era):
    """Every embedding nick of the era with a tau final state in the database, without 2016B ver1 (spec 3.1)."""
    sibling = json.loads(SIBLING_DATABASE.read_text())
    expected = sorted(n for n, entry in sibling.items() if entry["sample_type"] == "embedding" and str(entry["era"]) == era
                      and any(token in n for token in ("_eltau_", "_mutau_", "_tautau_")) and "_B_ver1_" not in n)
    assert (INVENTORY / f"{list_name('embedding', era)}.txt").read_text() == "".join(f"{n}\n" for n in expected)


def test_the_2018_embedding_inventories_are_one_list():
    # every 2018 embedding nick with a tau final state, one per line (the v5/v6 copy and the v7 copy of CROWN sm_2018_embedding.txt)
    assert (INVENTORY / f"{EMBEDDING_LIST}.txt").read_bytes() == (INVENTORY / f"{list_name('embedding', '2018')}.txt").read_bytes()


@pytest.mark.parametrize("era", ERAS)
def test_the_tau_id_list_of_an_era(era):
    """E17: W 0J/1J/2J instead of the inclusive W; SingleMuon data, the muon embedding mutau and muemb."""
    entries, nicks = database(), v7_nicks("tau_id", era)
    assert sorted(n for n in nicks if n.startswith("WJetsToLNu")) == sorted(nick(f"WJetsToLNu_{j}J_", era) for j in (0, 1, 2))
    assert {n.split("_Run")[0] for n in nicks if entries[n]["sample_type"] == "data"} == {"SingleMuon"}
    embedded = [n for n in nicks if entries[n]["sample_type"] == "embedding"]
    assert len(embedded) == 2 * PERIODS[era] and all(("_mutau_" in n) != ("_muemb_" in n) for n in embedded)


def test_the_borrowed_lists():
    """3.3: 2016postVFP takes VH -> tautau from 2016preVFP; the other eras borrow nothing."""
    assert (INVENTORY / f"{list_name('borrowed', '2016postVFP')}.txt").read_text() == "".join(
        f"2016preVFP/{nick(prefix, '2016preVFP')}\n" for prefix in sorted(VH_TAUTAU, key=lambda p: nick(p, "2016preVFP")))
    for era in ("2016preVFP", "2017", "2018"):
        assert (INVENTORY / f"{list_name('borrowed', era)}.txt").read_text() == ""


@pytest.mark.parametrize("era", ERAS)
def test_btag_components_exist_in_every_channel(era):
    for channel in TAU_CHANNELS + DILEPTON_CHANNELS:
        for component in BTAG_COMPONENTS:
            assert {f"btag_weight_upart_up_{component}", f"btag_weight_upart_down_{component}"} <= branches(channel, era), (channel, component)
    assert len(BTAG_COMPONENTS) == 4  # b/c and light, correlated and uncorrelated (WPs [T, M], C14)


@pytest.mark.parametrize("channel", TAU_CHANNELS + DILEPTON_CHANNELS)
def test_the_generated_2018_fixture_is_the_v2_file_plus_the_v6_columns(channel):
    """The branch fixtures come from CROWN's generated code; for 2018 ttbar they are the branches of a real
    sm2018_binned_v2 file (dump_branches.py) plus the columns v6 added (B7)."""
    v2 = {line.strip() for line in (BRANCHES / "sm2018_binned_v2" / f"{channel}.txt").read_text().splitlines() if line.strip()}
    v6 = {"npv_good", "rho", "jet_hadronFlavour_1", "jet_hadronFlavour_2", "bpair_hadronFlavour_1", "bpair_hadronFlavour_2"}
    assert fixture_branches(channel, "2018") == v2 | v6 and not v2 & v6
