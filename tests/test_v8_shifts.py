"""The shift model of the v8 production (v8 spec 6, SH16-SH28; switch shift_set v8): every shift of the n-tuples is
declared for the samples that have it (the inventory of the CROWN tests), the nuisances have the names and correlations
of the table of spec 6.1, the sums are those of their part shifts, the recoil nuisances sit on DY and W of 2018 alone, a
friend without a kinematic shift stops the skim, and the shifts without nuisance are declared but never filled."""
import dataclasses
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapesmith.datacards import write_datacard
from shapesmith.estimates import run_estimates
from shapesmith.events import select
from shapesmith.expressions import product
from shapesmith.fill import bookings, plan_booking, targets
from shapesmith.histogram import NOMINAL_VARIATION, HistKey, Histogram, HistogramSet, is_part, part_of
from shapesmith.model import AnalysisError, ColumnVariation, QuadratureSum, VariationSum, WeightVariation, declared_only
from shapesmith.ntuples import NtupleFile
from shapesmith.skim import check_shifts, column_variations, needed_columns, read_variations, required_shifted, skim_one
from shapesmith.testing import make_ntuple

from bbtautau_shapesmith import crown_names as crown
from bbtautau_shapesmith.analysis import friend_columns, sample_groups
from bbtautau_shapesmith.constants import ERAS, FF_SHIFTS_LT, FF_SHIFTS_TT, FF_SHIFTS_W, RECOIL_ERAS, TAU_CHANNELS, TAU_LEGS
from bbtautau_shapesmith.processes import EMBEDDED, SIGNAL
from bbtautau_shapesmith.switches import TauSwitches
from bbtautau_shapesmith.systematics import (
    EMBEDDING_OWN, EMBEDDING_SHARED, RECOIL_GROUPS, TAU_HIGHPT, TTBAR_THEORY, crown_shifts_v8, tau_highpt_variations, v8_variation_sums, v8_variations,
)
from bbtautau_shapesmith.theory_norms import RUNS_NORM
from tests.helpers import FIXTURES, REPO, sample_types
from tests.test_run2 import run2, tau  # noqa: F401 (the fixture)

INVENTORY = json.loads(gzip.open(FIXTURES / "v8_shift_inventory.json.gz").read())
SPEC_INVENTORY = REPO.parents[1] / "docs" / "analysen" / "v8" / "v8_spec_files" / "shift_inventory_final.json"
CHANNELS = ("et", "mt", "tt", "em")
DIRECTIONS = ("Up", "Down")
DMS = (0, 1, 10, 11)
FF_SUFFIXES = {f"__{key}{d}" for key in FF_SHIFTS_TT + FF_SHIFTS_W for d in DIRECTIONS}  # the shifts of the fake-factor friend
NN = ("nn_m3abnv1_predicted_class", "nn_m3abnv1_predicted_max_value")
V8 = {"shift_set": "v8", "crown_shifts": True}


def v8(database, era, **switches):
    """The tau analysis of an era on the v8 production: embedding in every tau channel, fake factors, the shifts as nuisances."""
    return tau(database, era, **{**V8, **switches})


def inventory_suffixes(era: str, sample_type: str, channel: str) -> set[str]:
    """The suffixes of the shifts the n-tuples of a sample type have in a scope (the inventory)."""
    families = INVENTORY["pairs"].get(f"{era}/{sample_type}", {}).get(channel, {})
    suffixes = {f"__{pair}{d}" for pairs in families.values() for pair in pairs for d in DIRECTIONS}
    return suffixes


def crown_variations(channel) -> list[ColumnVariation]:
    """The column variations of the channel that read CROWN shifts of the main n-tuples (not those of the fake-factor friend)."""
    return [v for v in channel.variations if isinstance(v, ColumnVariation) and v.suffix and v.suffix not in FF_SUFFIXES]


def nuisances(channel) -> set[str]:
    """The datacard nuisances the CROWN shifts of a channel make: the filled Up/Down pairs and the sums of parts."""
    filled = {v.name[: -len("Up")] for v in crown_variations(channel) if not declared_only(v) and not is_part(v.name) and v.name.endswith("Up")}
    return filled | {e.name for e in channel.estimators if isinstance(e, VariationSum)}


def expected_nuisances(channel: str, era: str, embedded: bool) -> set[str]:
    """The table of v8 spec 6.1 for a channel and era, written out."""
    copies_2018 = {"vse": ("2016preVFP", "2016postVFP"), "vsmu": ("2016preVFP", "2016postVFP", "2017")}  # K6: the payload is a copy of 2018
    vse, vsmu = ("2018" if era in copies_2018[family] else era for family in ("vse", "vsmu"))
    names = {f"CMS_scale_j_{source}" for source in ("Absolute", "BBEC1", "EC2", "FlavorQCD", "HF", "RelativeBal")}
    names |= {f"CMS_scale_j_{source}_{era}" for source in ("Absolute", "BBEC1", "EC2", "HF", "RelativeSample")}
    names |= {f"CMS_res_j_{era}", f"CMS_scale_met_unclustered_energy_{era}", "CMS_scale_e_13TeV", "CMS_res_e_13TeV", "CMS_pileup"}
    if era == "2018":
        names |= {"CMS_htt_boson_scale_met_2018", "CMS_htt_boson_res_met_2018"}
    if channel == "em":
        return names
    names |= {f"CMS_scale_t_dm{dm}_{era}" for dm in DMS}
    names |= {f"CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat{n}_DM{dm}_{era}" for n in (1, 2) for dm in DMS}
    names |= {"CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_alleras", f"CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_{era}"}
    names |= {f"CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_TES_dm{dm}_{era}" for dm in DMS}
    names |= {f"CMS_fake_e_dm{dm}_{region}_{vse}" for dm in DMS for region in ("barrel", "endcap")}
    names |= {f"CMS_fake_m_WH{wheel}_{vsmu}" for wheel in range(1, 6)}
    if channel == "et":
        names |= {f"CMS_scale_t_DeepTau2018v2p5_DM{dm}_genElectron_{region}_{vse}" for dm in DMS for region in ("barrel", "endcap")}  # the e -> tau ES is a 2018 copy in 2016 too
    if channel == "mt":
        names |= {f"CMS_scale_t_DeepTau2018v2p5_DM{dm}_genMuon_{era}" for dm in DMS}
    if channel == "tt":
        names |= {f"CMS_eff_trigger_tt_dm{dm}_{era}" for dm in DMS}
    if embedded:
        wp = "vsEleTight" if channel == "et" else "vsEleVVLoose"
        names |= {f"CMS_{kind}_t_emb_dm{token}_{wp}_{era}" for kind in ("scale", "eff") for token in ("0", "1", "1011")}
        if channel == "et":
            names |= {f"CMS_scale_e_barrel_emb_{era}", f"CMS_scale_e_endcap_emb_{era}"}
    return names


# ---- the declared shifts are those of the n-tuples (the inventory) ----


@pytest.mark.skipif(not SPEC_INVENTORY.exists(), reason="the shift inventory of the v8 spec is not on this machine")
def test_the_fixture_is_the_inventory_of_the_spec():
    """tests/fixtures/v8_shift_inventory.json.gz is scripts/make_shift_inventory_fixture.py on the file of the spec."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_shift_inventory_fixture", REPO / "scripts" / "make_shift_inventory_fixture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.compact(json.loads(SPEC_INVENTORY.read_text()), SPEC_INVENTORY.name) == INVENTORY


def test_the_inventory_has_every_era_sample_type_and_scope():
    surfaces = INVENTORY["pairs"]
    assert len(surfaces) == 64 and {key.split("/")[0] for key in surfaces} == set(ERAS)
    assert {key.split("/")[1] for key in surfaces} == set(sample_types().values()) | {"embedding"}
    assert all(set(scopes) == (set(TAU_CHANNELS) if key.endswith("/embedding") else set(CHANNELS)) for key, scopes in surfaces.items())
    counts = {scope: len([pair for pairs in surfaces["2018/ttbar"][scope].values() for pair in pairs]) for scope in CHANNELS}
    assert counts == {"et": 66, "mt": 65, "tt": 60, "em": 32}  # the pairs of the final CROWN inventory (f0d2b9b)
    assert {scope: sum(len(pairs) for pairs in families.values()) for scope, families in surfaces["2018/embedding"].items()} == {"et": 10, "mt": 10, "tt": 8}
    # the parton shower (C23) on every simulated sample, nowhere else
    for key, scopes in surfaces.items():
        for families in scopes.values():
            shower = {pair for pairs in families.values() for pair in pairs if pair.startswith("ps_")}
            assert shower == (set() if key.endswith(("/embedding", "/data")) else {"ps_isr", "ps_fsr"}), key


@pytest.mark.parametrize("era", ERAS)
@pytest.mark.parametrize("jet_fakes", ["ff", "mc"])
@pytest.mark.parametrize("crown_shifts", [True, False])
def test_every_sample_declares_exactly_the_shifts_of_its_ntuples(run2, era, jet_fakes, crown_shifts):
    """Per era, channel and sample (so per process group): the declared suffixes are those of the inventory for the
    sample type of the sample, in the era of its n-tuples, plus the parton shower; none is missing (the skim fails on
    an undeclared shift) and none is declared that the n-tuples lack (the skim fails on that too). With crown_shifts
    off they are declared all the same, without regions."""
    analysis = v8(run2, era, jet_fakes=jet_fakes, crown_shifts=crown_shifts)
    checked = set()
    for name, channel in analysis.channels.items():
        for sample in channel.samples:
            sample_type = sample_types()[sample.nick]
            declared = {v.suffix for v in column_variations(channel, sample) if v.suffix} - FF_SUFFIXES
            expected = inventory_suffixes(sample.era or era, sample_type, name)
            assert declared == expected, (name, sample.nick, sorted(declared ^ expected)[:6])
            checked.add((name, sample.group, sample_type))
            if not crown_shifts:
                assert all(declared_only(v) for v in column_variations(channel, sample) if v.suffix not in FF_SUFFIXES)
        assert {v.suffix for v in channel.variations if isinstance(v, ColumnVariation) and v.suffix in FF_SUFFIXES} == (
            set() if name == "em" or jet_fakes == "mc" else {f"__{key}{d}" for key in (FF_SHIFTS_TT if name == "tt" else FF_SHIFTS_LT) for d in DIRECTIONS})
    groups = {(name, group) for name, group, _ in checked}
    assert {group for name, group in groups if name == "mt"} >= {"data", SIGNAL, EMBEDDED, "DY", "TT", "ST", "VV", "TTV", "EWK", "ggH", "qqH", "ttH", "VH", "VH_htt"} | ({"W"} if jet_fakes == "mc" else set())
    assert {sample_type for _, _, sample_type in checked} == set(sample_types().values())  # every sample type of the lists (W: in em, and as MC jet fakes)
    assert ("W" in {group for name, group in groups if name != "em"}) == (jet_fakes == "mc")


def test_a_sample_borrowed_from_another_era_reads_the_shifts_of_its_ntuples(run2):
    """2016postVFP borrows VH -> tautau from 2016preVFP: its n-tuples name the shifts with 2016preVFP, the nuisances
    are those of the run's era, so that the process moves with the rest of 2016postVFP."""
    analysis = v8(run2, "2016postVFP")
    channel = analysis.channel("mt")
    borrowed = [s for s in channel.samples if s.era is not None]
    assert borrowed and {s.group for s in borrowed} == {"VH_htt"} and {s.era for s in borrowed} == {"2016preVFP"}
    own = next(s for s in channel.samples if s.group == "TT")
    for sample, token in ((borrowed[0], "2016preVFP"), (own, "2016postVFP")):
        variations = {v.name: v for v in column_variations(channel, sample)}
        assert variations["CMS_scale_j_Absolute_2016postVFPUp"].suffix == f"__CMS_scale_j_Absolute_{token}Up"
        assert variations["CMS_scale_t_dm1_2016postVFPDown"].suffix == f"__CMS_scale_t_DeepTau2018v2p5_DM1_genTau_{token}Down"
        assert variations[part_of("CMS_res_j_2016postVFPUp", "eta0to1p93")].suffix == f"__CMS_res_j_eta0to1p93_{token}Up"
        assert variations["CMS_scale_j_FlavorQCDUp"].suffix == "__CMS_scale_j_FlavorQCDUp"  # no era in the name: one variation for both
        assert variations[f"CMS_eff_m_id_{token}Up"].regions == ()  # declared only, under its CROWN name
    assert sample_groups(channel.samples, "2016postVFP")["VH_htt"] == ("mc", "2016preVFP") and sample_groups(channel.samples, "2016postVFP")["TT"] == ("mc", "2016postVFP")
    mixed = (dataclasses.replace(own, era="2017"), own)
    with pytest.raises(AnalysisError, match="the samples of the group TT are of the kinds or eras"):
        sample_groups(mixed, "2016postVFP")


@pytest.mark.parametrize("era", ERAS)
@pytest.mark.parametrize("channel", CHANNELS)
def test_a_shift_is_kinematic_where_the_inventory_says_so(era, channel):
    kinds = {pair: INVENTORY["kinds"][family] for scopes in INVENTORY["pairs"].values() for family, pairs in scopes.get(channel, {}).items() for pair in pairs}
    shifts = crown_shifts_v8(channel, era)
    assert len({s.crown for s in shifts}) == len(shifts)
    for shift in shifts:
        if shift.crown.startswith("ps_"):
            assert not shift.kinematic and shift.nuisance is None
            continue
        assert shift.kinematic == (kinds[shift.crown] == "kinematic"), shift.crown


# ---- names and correlations (v8 spec 6.1, SH27) ----


@pytest.mark.parametrize("era", ERAS)
def test_the_nuisances_are_those_of_the_table(run2, era):
    analysis = v8(run2, era)
    for name, channel in analysis.channels.items():
        assert nuisances(channel) == expected_nuisances(name, era, embedded=name != "em"), name
    from_mc = v8(run2, era, embedding=["et", "mt"]).channel("tt")  # v7b: the genuine taus of tt from MC (E33)
    assert nuisances(from_mc) == expected_nuisances("tt", era, embedded=False)
    assert not [v for v in from_mc.variations if "emb" in v.name]


def test_the_correlation_of_every_family():
    """Per family the nuisance names of the four eras: one name where it is correlated, the 2018 name where the payload
    of an era is a copy of 2018 (K6), else one per era (the 2016 halves separate)."""
    def names(crown_name, channel="mt"):
        result = []
        for era in ERAS:
            (shift,) = [s for s in crown_shifts_v8(channel, era) if s.crown == crown_name(era)]
            result.append(shift.nuisance)
        return result

    def per_era(base):
        return [f"{base}_{era}" for era in ERAS]

    assert names(lambda era: crown.jes("FlavorQCD")) == ["CMS_scale_j_FlavorQCD"] * 4
    assert names(lambda era: crown.jes("BBEC1", era)) == per_era("CMS_scale_j_BBEC1")  # the CMS list names the era, not the JME key 2016APV
    assert names(lambda era: crown.jer("eta0to1p93", era)) == per_era("CMS_res_j")
    assert names(lambda era: crown.unclustered_energy(era)) == per_era("CMS_scale_met_unclustered_energy")
    assert names(lambda era: crown.electron_es("scale", era)) == ["CMS_scale_e_13TeV"] * 4 and names(lambda era: crown.electron_es("res", era)) == ["CMS_res_e_13TeV"] * 4
    assert names(lambda era: crown.pileup(era)) == ["CMS_pileup"] * 4
    assert names(lambda era: crown.tau_es(10, era)) == per_era("CMS_scale_t_dm10")
    assert names(lambda era: crown.muon_fake_es(0, era)) == per_era("CMS_scale_t_DeepTau2018v2p5_DM0_genMuon")
    assert names(lambda era: crown.electron_fake_es(1, "endcap", era), "et") == [f"CMS_scale_t_DeepTau2018v2p5_DM1_genElectron_endcap_{token}" for token in ("2018", "2018", "2017", "2018")]
    assert names(lambda era: crown.vsjet_stat(2, 11, era)) == per_era("CMS_eff_t_DeepTau2018v2p5_VSjet_dm_stat2_DM11")
    assert names(lambda era: crown.vsjet_syst_alleras()) == ["CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_alleras"] * 4
    assert names(lambda era: crown.vsjet_syst(era)) == per_era("CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst")
    assert names(lambda era: crown.vsjet_syst_tes(1, era)) == per_era("CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_TES_dm1")
    assert names(lambda era: crown.vsele(0, "barrel", era)) == [f"CMS_fake_e_dm0_barrel_{token}" for token in ("2018", "2018", "2017", "2018")]
    assert names(lambda era: crown.vsmu(3, era)) == ["CMS_fake_m_WH3_2018"] * 4
    assert names(lambda era: crown.ditau_trigger(10, era), "tt") == per_era("CMS_eff_trigger_tt_dm10")
    assert names(lambda era: crown.embedding_tau_es(11, era)) == per_era("CMS_scale_t_emb_dm1011_vsEleVVLoose")
    assert names(lambda era: crown.embedding_tau_id(0, era), "et") == per_era("CMS_eff_t_emb_dm0_vsEleTight")
    assert names(lambda era: crown.embedding_electron_es("barrel", era), "et") == per_era("CMS_scale_e_barrel_emb")  # V34: 2017 as the others
    (recoil,) = {s.nuisance for s in crown_shifts_v8("mt", "2018") if s.crown == crown.recoil("scale", "2018")}
    assert recoil == "CMS_htt_boson_scale_met_2018"
    assert [tau_highpt_variations("mt", era)[0].name for era in ERAS] == [f"CMS_eff_t_highpT_{era}Up" for era in ERAS]  # V35: uncorrelated between the eras


@pytest.mark.parametrize("era", ERAS)
def test_the_shifts_without_nuisance_are_declared_without_regions(run2, era):
    """V7, V25, SH26: lepton ID, isolation and trigger, the QCD scale, the parton shower and the muon energy scale."""
    analysis = v8(run2, era)
    expected = {
        "mt": {f"CMS_eff_m_id_{era}", f"CMS_eff_m_iso_{era}", f"CMS_eff_m_trigger_{era}", f"CMS_scale_m_etalt1p2_{era}", f"CMS_scale_m_eta1p2to2p1_{era}"},
        "et": {f"CMS_eff_e_id_{era}", f"CMS_eff_e_trigger_{era}"},
        "tt": set(),
        "em": {f"CMS_eff_m_id_{era}", f"CMS_eff_m_iso_{era}", f"CMS_eff_m_trigger_{era}", f"CMS_eff_e_id_{era}"} | {f"CMS_scale_m_{region}_{era}" for region in ("etalt1p2", "eta1p2to2p1", "eta2p1to2p4")},
    }
    borrowed = "2016preVFP" if era == "2016postVFP" else None  # VH -> tautau: its n-tuples name the shifts with their era
    for name, channel in analysis.channels.items():
        declared = {v.name[: -len("Up")]: v for v in crown_variations(channel) if declared_only(v) and v.name.endswith("Up")}
        other_era = {base.replace(era, borrowed) for base in expected[name]} if borrowed else set()
        assert set(declared) == expected[name] | other_era | {"QCDscale_ren", "QCDscale_fac", "ps_isr", "ps_fsr"}, name
        assert all(v.groups == ("VH_htt",) for base, v in declared.items() if base in other_era)
        assert all(v.suffix == f"__{base}Up" and v.required == () for base, v in declared.items())  # under the CROWN name
        muon_es = [v for base, v in declared.items() if base.startswith("CMS_scale_m_")]
        assert all(v.applies_to == (("mc", "embedding") if name == "mt" else ("mc",)) for v in muon_es)  # the embedding has it in mt
        assert not {v.name for v in crown_variations(channel) if not declared_only(v)} & {f"{base}{d}" for base in declared for d in DIRECTIONS}


# ---- sums of part shifts (V20, V23) ----


@pytest.mark.parametrize("era", ERAS)
def test_the_jer_is_the_sum_of_its_six_regions(run2, era):
    regions = ("eta0to1p93", "eta1p93to2p5", "eta2p5to3_ptlt50", "eta2p5to3_ptgt50", "eta3to5_ptlt50", "eta3to5_ptgt50")
    analysis = v8(run2, era)
    for name, channel in analysis.channels.items():
        assert VariationSum(f"CMS_res_j_{era}", regions) in channel.estimators
        parts = [v for v in channel.variations if is_part(v.name) and v.name.startswith("CMS_res_j")]
        assert {v.name for v in parts} == {part_of(f"CMS_res_j_{era}{d}", region) for region in regions for d in DIRECTIONS}
        for v in parts:  # the borrowed samples of 2016postVFP read the regions under the names of their era
            sample_era = "2016preVFP" if v.groups == ("VH_htt",) else era
            assert v.suffix == f"__CMS_res_j_{v.name.split('%')[1]}_{sample_era}{'Up' if 'Up%' in v.name else 'Down'}" and v.applies_to == ("mc",) and v.regions == ("nominal",)
        assert len(parts) == 12 * (2 if era == "2016postVFP" else 1)
    # the regions stay selectable: six nuisances and no sum
    split = v8(run2, era, jer_regions=True)
    for name, channel in split.channels.items():
        assert not [e for e in channel.estimators if isinstance(e, VariationSum) and e.name.startswith("CMS_res_j")]
        assert {n for n in nuisances(channel) if n.startswith("CMS_res_j")} == {f"CMS_res_j_{region}_{era}" for region in regions}
        assert nuisances(channel) - {f"CMS_res_j_{region}_{era}" for region in regions} == expected_nuisances(name, era, name != "em") - {f"CMS_res_j_{era}"}


def _hist(*values):
    return Histogram([0.0, 1.0, 2.0], list(values), [0.1, 0.1])


def _key(channel, process, variation=NOMINAL_VARIATION):
    return HistKey(channel, "nominal", process, "nominal", variation, "yield2")


def test_the_summed_nuisances_are_built_from_their_parts(run2):
    """Estimates on filled parts: the JER of every simulated process, the embedding tau ES with 0.866 of its shift in
    its own nuisance and 0.5 in the one of the simulation (V20), which the simulation carries as its own shift."""
    analysis = v8(run2, "2018", fit_categories="control")
    hset = HistogramSet()
    for process in ("data", "EMB", "TTL", "jetFakes"):
        hset[_key("mt", process)] = _hist(100.0, 200.0)
    for region, shift in (("eta0to1p93", (4.0, 0.0)), ("eta1p93to2p5", (0.0, -6.0)), ("eta3to5_ptgt50", (1.0, 1.0))):  # three of the six regions move TTL
        hset[_key("mt", "TTL", part_of("CMS_res_j_2018Up", region))] = _hist(100.0 + shift[0], 200.0 + shift[1])
        hset[_key("mt", "TTL", part_of("CMS_res_j_2018Down", region))] = _hist(100.0 - shift[0], 200.0 - shift[1])
    own = "CMS_scale_t_emb_dm{}_vsEleVVLoose_2018"
    for dm, token, shift in ((0, "0", 10.0), (1, "1", 20.0), (10, "1011", 30.0), (11, "1011", 40.0)):
        hset[_key("mt", "EMB", part_of(own.format(token) + "Up", f"dm{dm}"))] = _hist(100.0 + shift, 200.0)
        hset[_key("mt", "EMB", part_of(own.format(token) + "Down", f"dm{dm}"))] = _hist(100.0 - shift / 2, 200.0)
    hset[_key("mt", "TTL", "CMS_scale_t_dm0_2018Up")] = _hist(103.0, 200.0)  # the shift of the simulation itself
    hset[_key("mt", "TTL", "CMS_scale_t_dm0_2018Down")] = _hist(98.0, 200.0)
    run_estimates(hset, analysis, ["mt"])

    def values(process, variation):
        return hset[_key("mt", process, variation)].values.tolist()

    assert values("TTL", "CMS_res_j_2018Up") == pytest.approx([105.0, 195.0]) and values("TTL", "CMS_res_j_2018Down") == pytest.approx([95.0, 205.0])
    assert not hset.select(process="EMB", variation="CMS_res_j_2018Up")  # no part: the embedded sample has no JER shift
    assert (EMBEDDING_SHARED, EMBEDDING_OWN) == (0.5, 0.866) and EMBEDDING_SHARED**2 + EMBEDDING_OWN**2 == pytest.approx(1.0, abs=1e-4)
    assert values("EMB", own.format("0") + "Up") == pytest.approx([108.66, 200.0]) and values("EMB", own.format("0") + "Down") == pytest.approx([100 - 0.866 * 5, 200.0])
    assert values("EMB", own.format("1") + "Up") == pytest.approx([100 + 0.866 * 20, 200.0])
    assert values("EMB", own.format("1011") + "Up") == pytest.approx([100 + 0.866 * 70, 200.0])  # DM10 + DM11: one measured category
    for dm, shift in ((0, 10.0), (1, 20.0), (10, 30.0), (11, 40.0)):  # the nuisance of the simulation, per decay mode
        assert values("EMB", f"CMS_scale_t_dm{dm}_2018Up") == pytest.approx([100 + 0.5 * shift, 200.0])
        assert values("EMB", f"CMS_scale_t_dm{dm}_2018Down") == pytest.approx([100 - 0.25 * shift, 200.0])
    assert values("TTL", "CMS_scale_t_dm0_2018Up") == [103.0, 200.0] and values("TTL", "CMS_scale_t_dm0_2018Down") == [98.0, 200.0]
    assert all(hset[_key("mt", "EMB", own.format("0") + "Up")].variances == hset[_key("mt", "EMB")].variances)
    assert {key.variation for key in hset.select(process="jetFakes")} == {NOMINAL_VARIATION}  # V31: only the fake-factor shifts reach it


def test_the_embedding_sums_of_every_channel(run2):
    for channel in TAU_CHANNELS:
        wp = "vsEleTight" if channel == "et" else "vsEleVVLoose"
        sums = v8_variation_sums(channel, "2017", embedded=True)
        assert sums[0] == VariationSum("CMS_res_j_2017", crown.JER_REGIONS)
        scale = {e.name: e for e in sums[1:]}
        assert scale[f"CMS_scale_t_emb_dm0_{wp}_2017"] == VariationSum(f"CMS_scale_t_emb_dm0_{wp}_2017", ("dm0",), scale=0.866)
        assert scale[f"CMS_scale_t_emb_dm1011_{wp}_2017"] == VariationSum(f"CMS_scale_t_emb_dm1011_{wp}_2017", ("dm10", "dm11"), scale=0.866)
        assert scale["CMS_scale_t_dm11_2017"] == VariationSum("CMS_scale_t_dm11_2017", ("dm11",), scale=0.5, source=f"CMS_scale_t_emb_dm1011_{wp}_2017")
        assert scale[f"CMS_eff_t_emb_dm1011_{wp}_2017"] == VariationSum(f"CMS_eff_t_emb_dm1011_{wp}_2017", ("dm10", "dm11"))  # the vsJet ID: no share with the simulation
        assert len(sums) == 1 + 3 + 4 + 1
        assert v8_variation_sums(channel, "2017", embedded=False) == sums[:1] and v8_variation_sums(channel, "2017", True, jer_regions=True) == sums[1:]
    assert v8_variation_sums("em", "2017", embedded=True) == (VariationSum("CMS_res_j_2017", crown.JER_REGIONS),)


def test_the_embedding_vsjet_id_keeps_its_shifts_whole(run2):
    """Only the tau ES is shared with the simulation (V20): dm0 and dm1 of the vsJet ID are plain nuisances."""
    channel = v8(run2, "2018").channel("et")
    emb = {v.name: v for v in channel.variations if isinstance(v, ColumnVariation) and v.applies_to == ("embedding",)}
    assert emb["CMS_eff_t_emb_dm0_vsEleTight_2018Up"].suffix == "__CMS_eff_t_emb_DeepTau2018v2p5_VSjet_DM0_2018Up"
    assert emb[part_of("CMS_eff_t_emb_dm1011_vsEleTight_2018Down", "dm11")].suffix == "__CMS_eff_t_emb_DeepTau2018v2p5_VSjet_DM11_2018Down"
    assert emb[part_of("CMS_scale_t_emb_dm0_vsEleTight_2018Up", "dm0")].suffix == "__CMS_scale_t_emb_DeepTau2018v2p5_DM0_2018Up"
    assert "CMS_scale_t_emb_dm0_vsEleTight_2018Up" not in emb  # a part: the sums scale it
    assert emb["CMS_scale_e_endcap_emb_2018Down"].suffix == "__CMS_scale_e_endcap_emb_2018Down"
    assert all(v.regions == ("nominal",) for v in emb.values())


# ---- recoil and unclustered energy (V9) ----


@pytest.mark.parametrize("era", ERAS)
@pytest.mark.parametrize("jet_fakes", ["ff", "mc"])
def test_the_recoil_nuisances_sit_on_dy_and_w_of_2018_only(run2, era, jet_fakes):
    analysis = v8(run2, era, jet_fakes=jet_fakes)
    for name, channel in analysis.channels.items():
        carried = {}  # nuisance -> the processes that are booked with it
        for booking in bookings(channel, True):
            for variation in booking.variations:
                nuisance = variation.name.split("%")[0]  # of a part: the summed nuisance
                if not declared_only(variation) and nuisance.endswith("Up"):
                    carried.setdefault(nuisance[: -len("Up")], set()).add(booking.process.name)
        recoil = {process.name for process in channel.processes if process.group in RECOIL_GROUPS and process.role != "auxiliary"}
        simulated = {process.name for process in channel.processes if process.role in ("signal", "background") and channel.kind_of(process) == "mc"}
        assert recoil and recoil < simulated
        if era in RECOIL_ERAS:
            assert carried["CMS_htt_boson_scale_met_2018"] == carried["CMS_htt_boson_res_met_2018"] == recoil
            assert carried[f"CMS_scale_met_unclustered_energy_{era}"] == simulated - recoil  # and the reverse
        else:
            assert not [n for n in carried if "boson" in n] and carried[f"CMS_scale_met_unclustered_energy_{era}"] == simulated
        assert carried["CMS_scale_j_FlavorQCD"] == carried[f"CMS_res_j_{era}"] == carried["CMS_pileup"] == simulated
        assert {"EWK", "EWKL", "EWKT", "EWKJ"} & simulated <= carried[f"CMS_scale_met_unclustered_energy_{era}"]  # EWK W/Z has no recoil correction
    assert RECOIL_ERAS == ("2018",) and RECOIL_GROUPS == ("DY", "W")


def test_the_recoil_groups_are_the_recoil_corrected_sample_types():
    """DY and W are the groups of the CROWN sample types dyjets and wjets, the ones the inventory gives the recoil shifts."""
    recoil_types = {key.split("/")[1] for key, scopes in INVENTORY["pairs"].items() if any("recoil calibration" in families for families in scopes.values())}
    assert recoil_types == {"dyjets", "wjets"}
    assert {key.split("/")[0] for key, scopes in INVENTORY["pairs"].items() if any("recoil calibration" in families for families in scopes.values())} == set(RECOIL_ERAS)
    from bbtautau_shapesmith.samples import route

    by_group = {}
    for nick, sample_type in sample_types().items():
        if sample_type not in ("data", "embedding"):
            by_group.setdefault(route(nick)[0], set()).add(sample_type)
    assert by_group["DY"] == {"dyjets"} and by_group["W"] == {"wjets"}
    assert not [group for group, types in by_group.items() if types & recoil_types and group not in RECOIL_GROUPS]


# ---- friends must follow the kinematic shifts (SH19, V31) ----


def _friend_run(run2, **switches):
    return v8(run2, "2018", channels=("mt",), embedding=["mt"], nn_friend=True, nn_prefix="nn_m3abnv1_", fastmtt_friend=True, fit_categories="nn", **switches)


def test_the_kinematic_shifts_require_the_friend_columns(run2):
    analysis = _friend_run(run2)
    channel = analysis.channel("mt")
    friends = friend_columns("mt", TauSwitches(sample_lists=["x"], jet_fakes="ff", nn_friend=True, nn_prefix="nn_m3abnv1_", fastmtt_friend=True))
    assert set(NN) <= set(friends) and {"nn_m3abnv1_nn_score_HH2B2Tau", "m_fastmtt", "pt_fastmtt", "fake_factor", "fake_factor_1", "fake_factor_raw"} <= set(friends)
    kinds = {pair: INVENTORY["kinds"][family] for family, pairs in INVENTORY["pairs"]["2018/ttbar"]["mt"].items() for pair in pairs}
    kinds.update({pair: INVENTORY["kinds"][family] for family, pairs in INVENTORY["pairs"]["2018/embedding"]["mt"].items() for pair in pairs})
    kinds.update({pair: INVENTORY["kinds"][family] for family, pairs in INVENTORY["pairs"]["2018/dyjets"]["mt"].items() for pair in pairs})
    for variation in crown_variations(channel):
        pair = variation.suffix[2:].removesuffix("Up").removesuffix("Down")
        kinematic = kinds.get(pair) == "kinematic"
        assert variation.required == (friends if kinematic and not declared_only(variation) else ()), variation.name
    assert friend_columns("em", TauSwitches(sample_lists=["x"], nn_friend=True, fastmtt_friend=True)) == ()  # em reads no friend
    ttbar = next(s for s in channel.samples if s.group == "TT")
    required = required_shifted(channel, ttbar)
    # where the shift is filled (the nominal region) the categories read the class and the score, the control variables FastMTT
    assert required["CMS_scale_j_AbsoluteUp"] == sorted(f"{column}__CMS_scale_j_AbsoluteUp" for column in (*NN, "m_fastmtt", "pt_fastmtt", "eta_fastmtt", "phi_fastmtt"))
    assert not [column for columns in required.values() for column in columns if column.startswith("fake_factor")]  # V31: the anti-isolated region books none
    assert "CMS_pileupUp" not in required and "CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_2018Up" not in required  # weights leave the friends alone
    assert len(required) == 2 * (11 + 6 + 1 + 2 + 4 + 4)  # JES, JER regions, unclustered energy, electron ES, tau ES, mu -> tau ES
    data = next(s for s in channel.samples if s.kind == "data")
    assert required_shifted(channel, data) == {}


def _branches(channel, sample, friend_shifted=()) -> tuple[set[str], set[str]]:
    """The needed columns of a sample and synthetic branches of its n-tuple with every declared shift: each main
    column under each shift, the friend columns nominal and, of `friend_shifted`, also shifted."""
    columns = needed_columns(channel, sample)
    friends = {c for c in columns if c.startswith(("nn_", "fake_factor")) or c.endswith("_fastmtt")}
    suffixes = {v.suffix for v in column_variations(channel, sample) if v.suffix}
    main = {c + suffix for c in columns - friends for suffix in suffixes - FF_SUFFIXES}
    ff = {c + suffix for c in friends if c.startswith("fake_factor") for suffix in suffixes & FF_SUFFIXES}
    return columns, columns | main | ff | {c + suffix for c in friend_shifted for suffix in suffixes - FF_SUFFIXES}


def test_a_friend_without_a_kinematic_shift_stops_the_skim(run2):
    analysis = _friend_run(run2)
    channel = analysis.channel("mt")
    ttbar = next(s for s in channel.samples if s.group == "TT")
    nn_and_fastmtt = (*NN, "m_fastmtt", "pt_fastmtt", "eta_fastmtt", "phi_fastmtt")
    columns, branches = _branches(channel, ttbar, nn_and_fastmtt)
    check_shifts(channel, ttbar, columns, branches)  # NN and FastMTT friends with every shift; the fake-factor friend nominal only
    for missing, column in (("nn_m3abnv1_predicted_class__CMS_scale_j_EC2_2018Down", "nn_m3abnv1_predicted_class"), ("m_fastmtt__CMS_scale_t_DeepTau2018v2p5_DM10_genTau_2018Up", "m_fastmtt"),
                            (f"nn_m3abnv1_predicted_max_value__{crown.jer('eta2p5to3_ptlt50', '2018')}Up", "nn_m3abnv1_predicted_max_value")):
        shift = missing.split("__")[1]
        with pytest.raises(ValueError, match=rf"\(__{shift}\): the required column {column} has no shifted branch"):
            check_shifts(channel, ttbar, columns, branches - {missing})
    # a friend produced without shifts: every kinematic shift is reported, no weight shift
    columns, nominal_friends = _branches(channel, ttbar)
    with pytest.raises(ValueError) as excinfo:
        check_shifts(channel, ttbar, columns, nominal_friends)
    problems = str(excinfo.value).split("; ")
    assert len(problems) == 2 * (11 + 6 + 1 + 2 + 4 + 4) * 6 and all("the required column" in problem for problem in problems)
    assert not [problem for problem in problems if "VSjet" in problem or "pileup" in problem or "fake_factor" in problem or "CMS_scale_m_" in problem]
    emb = next(s for s in channel.samples if s.kind == "embedding")
    columns, nominal_friends = _branches(channel, emb)
    with pytest.raises(ValueError) as excinfo:
        check_shifts(channel, emb, columns, nominal_friends)
    assert {problem.split("(__")[1].split(")")[0].removesuffix("Up").removesuffix("Down") for problem in str(excinfo.value).split("; ")} == {f"CMS_scale_t_emb_DeepTau2018v2p5_DM{dm}_2018" for dm in DMS}
    # declared only (crown_shifts off), nothing is required: the S0 cards need no shifted friend
    s0 = _friend_run(run2, crown_shifts=False).channel("mt")
    ttbar = next(s for s in s0.samples if s.group == "TT")
    columns, nominal_friends = _branches(s0, ttbar)
    check_shifts(s0, ttbar, columns, nominal_friends)
    assert required_shifted(s0, ttbar) == {}


# ---- the skim of a synthetic v8 n-tuple ----


def _ntuple(tmp_path, channel, sample, shifted_friend: bool, n=40) -> NtupleFile:
    """A main file with every needed n-tuple column of the sample and, for each declared shift, pt_2 and lhe_scale_weight
    shifted; a friend file with the NN, FastMTT and fake-factor columns (with the fake-factor shifts; with
    `shifted_friend` the NN and FastMTT columns under every other shift too)."""
    rng = np.random.default_rng(1)
    columns = needed_columns(channel, sample)
    friends = {c for c in columns if c.startswith(("nn_", "fake_factor")) or c.endswith("_fastmtt")}
    integers = {"event", "run", "lumi"}

    def column(name):
        if name in integers:
            return np.arange(n, dtype=np.uint64)
        if "predicted_class" in name:
            return np.zeros(n, dtype=np.int32)
        if name.startswith(("pt_", "m_", "bpair_pt", "mt_")):
            return rng.uniform(30.0, 60.0, n).astype(np.float32)
        return np.ones(n, dtype=np.float32)

    main = {name: column(name) for name in sorted(columns - friends)}
    main["lhe_scale_weight"] = np.ones(n, dtype=np.float32)
    friend = {name: column(name) for name in sorted(friends)}
    for variation in column_variations(channel, sample):
        if variation.suffix in FF_SUFFIXES:
            friend.update({c + variation.suffix: column(c) for c in friends if c.startswith("fake_factor")})
            continue
        main["pt_2" + variation.suffix] = (main["pt_2"] * 1.01).astype(np.float32)
        main["lhe_scale_weight" + variation.suffix] = main["lhe_scale_weight"]
        if shifted_friend:
            friend.update({c + variation.suffix: column(c) for c in friends if not c.startswith("fake_factor")})
    directory = tmp_path / ("shifted" if shifted_friend else "nominal")
    make_ntuple(directory / "main.root", main)
    make_ntuple(directory / "friend.root", friend)
    return NtupleFile(str(directory / "main.root"), (str(directory / "friend.root"),), sample.nick, channel.name, "main.root")


@pytest.mark.parametrize("group", ["TT", EMBEDDED])
def test_the_skim_of_a_v8_ntuple(run2, tmp_path, group):
    """End to end on a synthetic file of one sample: with shifted friends the skim holds the filled shifts and none of
    the declared-only ones; a friend without the kinematic shifts stops it; with crown_shifts off (S0) the same file is
    skimmed with nominal friends and holds no CROWN shift at all."""
    channel = _friend_run(run2).channel("mt")
    sample = next(s for s in channel.samples if s.group == group)
    columns = needed_columns(channel, sample)
    result = skim_one(_ntuple(tmp_path, channel, sample, shifted_friend=True), sample, channel, columns, tmp_path / "skim" / "shifted.parquet")
    skim = pd.read_parquet(result.path)
    shifted = {name.split("__", 1)[1] for name in skim.columns if "__" in name}
    filled = {v.suffix[2:] for v in read_variations(channel, sample) if v.suffix}
    assert shifted == filled and result.n_in == 40
    assert not [name for name in shifted if name.startswith(("CMS_scale_m_", "CMS_eff_m_", "QCDscale", "ps_"))]  # declared only: not read
    kinematic = "CMS_scale_t_emb_DeepTau2018v2p5_DM0_2018Up" if group == EMBEDDED else "CMS_scale_j_AbsoluteUp"
    assert {f"pt_2__{kinematic}", f"nn_m3abnv1_predicted_class__{kinematic}", f"m_fastmtt__{kinematic}"} <= set(skim.columns)
    with pytest.raises(ValueError, match=rf"declared shift .*\(__{kinematic}\): the required column m_fastmtt has no shifted branch"):
        skim_one(_ntuple(tmp_path, channel, sample, shifted_friend=False), sample, channel, columns, tmp_path / "skim" / "nominal.parquet")
    s0 = _friend_run(run2, crown_shifts=False).channel("mt")
    result = skim_one(_ntuple(tmp_path, s0, sample, shifted_friend=False), sample, s0, needed_columns(s0, sample), tmp_path / "skim" / "s0.parquet")
    s0_shifted = {name.split("__", 1)[1] for name in pd.read_parquet(result.path).columns if "__" in name}
    assert s0_shifted == {key + d for key in FF_SHIFTS_LT for d in DIRECTIONS}  # the fake-factor friend shifts, as in v7b


# ---- declared shifts without nuisance never reach the card (SH26) ----


def _filled(analysis, channels) -> HistogramSet:
    """What `hist` would fill for the categories of the channels (fill.plan_booking on a schema with every shifted
    branch), each as a two-bin histogram: nominal 10 per bin (the data 1000, for a positive jet-fake estimate), Up 10 %
    more and Down 10 % less in the first bin."""
    hset = HistogramSet()
    for name in channels:
        channel = analysis.channel(name)
        for booking in bookings(channel, True):
            sample = channel.samples_of(booking.process.group)[0]
            _, available = _branches(channel, sample, friend_shifted=NN)
            for fill in plan_booking(channel, booking, targets(channel, False, None), available):
                for target, _, _ in fill.targets:
                    variation = fill.variation.name if fill.variation else NOMINAL_VARIATION
                    scale = 100.0 if booking.process.role == "data" else 1.0
                    first = 11.0 if variation.endswith("Up") or "Up%" in variation else 9.0 if variation.endswith("Down") or "Down%" in variation else 10.0 if variation == NOMINAL_VARIATION else 10.5
                    hset[HistKey(name, target.category, booking.process.name, fill.region, variation, target.variable.name)] = Histogram([0.0, 1.0, 2.0], [scale * first, scale * 10.0], [0.1, 0.1])
    return hset


def _card(card):
    text = card.read_text()
    rows = {line.split()[0]: line.split() for line in text.splitlines() if " lnN " in line or " shape " in line}
    return text, {name: row[1] for name, row in rows.items()}, rows


@pytest.mark.parametrize("era", ["2018", "2016postVFP"])
def test_the_card_has_the_nuisances_and_none_of_the_declared_only_shifts(run2, tmp_path, era):
    """hist plan -> estimates -> card for every channel: the shape lines are the nuisances of the table (with the
    fake-factor and nominal-tree shapes of before), and no declared-only shift is filled or written."""
    analysis = v8(run2, era, fit_categories="control", study_uncertainties=True, embedding=["et", "mt"], signal_theory=["qcd_scale", "parton_shower", "pdf"], ttbar_shape=True,
                  ttbar_parton_shower=True)
    hset = _filled(analysis, CHANNELS)
    declared = {v.name for name in CHANNELS for v in crown_variations(analysis.channel(name)) if declared_only(v)}
    assert declared and not {key.variation for key in hset} & declared  # never filled
    assert not [key for key in hset if key.region != "nominal" and key.variation != NOMINAL_VARIATION and not key.variation.startswith(("CMS_ff_", "CMS_btag", "CMS_prefiring"))]  # V31
    run_estimates(hset, analysis, list(CHANNELS))
    text, kinds, rows = _card(write_datacard(hset, analysis, list(CHANNELS), tmp_path / "all", min_background=None))
    shapes = {name for name, kind in kinds.items() if kind == "shape"}
    crown_shapes = set().union(*(expected_nuisances(name, era, embedded=name in ("et", "mt")) for name in CHANNELS))
    theory = {"QCDscale_ren_HH_acc", "QCDscale_fac_HH_acc", "ps_isr_HH", "ps_fsr_HH", "pdf_HH_acc", "ps_isr_ttbar", "ps_fsr_ttbar", "CMS_htt_ttbarShape"}
    highpt = {f"CMS_eff_t_highpT_{era}"}
    others = {name for name in shapes if name.startswith(("CMS_ff_", "CMS_btag_", "CMS_htt_emb_ttbar"))}
    assert shapes - others == crown_shapes | theory | highpt
    assert {name for name in shapes if name.startswith("CMS_ff_dm_nonclosure")} and {name for name in others if name.startswith("CMS_btag_")}  # they stay
    for name in ("CMS_scale_m_", "CMS_eff_m_id", "CMS_eff_m_iso", "CMS_eff_e_id", "QCDscale_ren ", "QCDscale_fac ", "ps_isr ", "ps_fsr "):
        assert name not in text, name
    assert "%" not in text  # no part of a sum
    # V7: the lepton ID and trigger stay the lnN they were; the declared-only trigger shift of the same name adds no shape
    assert kinds["eff_m"] == kinds["eff_e"] == kinds[f"CMS_eff_m_trigger_{era}"] == kinds[f"CMS_eff_e_trigger_{era}"] == kinds[f"CMS_eff_m_trigger_emb_{era}"] == "lnN"
    assert kinds["eff_t_vsLep_mt_2018" if era != "2017" else "eff_t_vsLep_mt_2017"] == "lnN"  # S10: genuine taus keep it
    # SH17: the stand-ins of the study are gone where a shift took their place, the other lnN of the study stay
    assert kinds[f"CMS_scale_t_dm0_{era}"] == "shape" and not [name for name in kinds if name.startswith(("CMS_eff_t_vsJet", "eff_t_vsJet", "CMS_eff_t_trigger_ditau"))]
    assert kinds[f"CMS_scale_t_emb_dm0_vsEleTight_{era}"] == "shape" and kinds[f"CMS_eff_t_emb_dm1011_vsEleVVLoose_{era}"] == "shape"
    assert kinds[f"CMS_eff_m_stat_{era}"] == kinds["CMS_eff_e_reco_13TeV"] == "lnN" and "CMS_eff_e_reco " not in text  # SH27
    assert kinds.get(f"CMS_htt_met_mt_acceptance_{era}") == ("lnN" if era != "2018" else None)
    if era == "2016postVFP":
        assert kinds["CMS_ff_2016half_mt_2016postVFP"] == "lnN" and kinds["CMS_htt_borrowed_VH_htt_2016postVFP"] == "lnN"
    # the signal carries the theory shapes and the shifts; the embedded sample the tau ES of the simulation, scaled
    lines = text.splitlines()
    bins = lines[[i for i, line in enumerate(lines) if line.startswith("bin ")][1]].split()[1:]
    processes = lines[[i for i, line in enumerate(lines) if line.startswith("process")][0]].split()[1:]
    column = {(b, p): i + 2 for i, (b, p) in enumerate(zip(bins, processes))}
    mt, tt, em = f"htt_mt_1_{era}", f"htt_tt_1_{era}", f"htt_em_1_{era}"
    assert rows["pdf_HH_acc"][column[mt, SIGNAL]] == rows["ps_isr_HH"][column[em, SIGNAL]] == "1" and rows["pdf_HH_acc"][column[mt, "TTL"]] == "-"
    assert rows["ps_fsr_ttbar"][column[mt, "TTL"]] == rows["ps_fsr_ttbar"][column[em, "TT"]] == rows["ps_fsr_ttbar"][column[tt, "TTT"]] == "1" and rows["ps_fsr_ttbar"][column[mt, SIGNAL]] == "-"
    assert rows[f"CMS_scale_t_dm1_{era}"][column[mt, EMBEDDED]] == rows[f"CMS_scale_t_dm1_{era}"][column[mt, SIGNAL]] == "1" and rows[f"CMS_scale_t_dm1_{era}"][column[em, "TT"]] == "-"
    assert rows[f"CMS_res_j_{era}"][column[mt, EMBEDDED]] == rows[f"CMS_res_j_{era}"][column[mt, "jetFakes"]] == "-" and rows[f"CMS_res_j_{era}"][column[em, "TT"]] == "1"
    assert rows[f"CMS_eff_t_highpT_{era}"][column[mt, "TTL"]] == rows[f"CMS_eff_t_highpT_{era}"][column[tt, "ZTT"]] == "1" and rows[f"CMS_eff_t_highpT_{era}"][column[mt, EMBEDDED]] == "-"
    assert rows[f"CMS_eff_trigger_tt_dm0_{era}"][column[tt, "ZTT"]] == "1" and rows[f"CMS_eff_trigger_tt_dm0_{era}"][column[mt, "TTL"]] == "-"
    if era == "2016postVFP":  # the borrowed process moves with the nuisances of the run's era
        assert rows["CMS_scale_j_Absolute_2016postVFP"][column[mt, "VH_htt"]] == rows["CMS_res_j_2016postVFP"][column[mt, "VH_htt"]] == "1" and "2016preVFP" not in text


# ---- the tau_h above 140 GeV (V35) ----


def _factor(channel, process, name, frame):
    """The weight of the process under the weight variation over its nominal weight."""
    variation = next(v for v in channel.variations if v.name == name)
    definition = channel.process(process)
    nominal, varied = (select(channel, definition, channel.region("nominal"), v)[1] for v in (None, variation))
    return product(frame, varied) / product(frame, nominal)


def test_the_highpt_shape_is_the_envelope_of_both_sets_above_140_gev(run2):
    analysis = v8(run2, "2018", channels=("mt", "tt"), embedding=["mt"])
    mt = analysis.channel("mt")
    up, down = (v for v in mt.variations if v.name.startswith(TAU_HIGHPT))
    assert (up.name, down.name) == ("CMS_eff_t_highpT_2018Up", "CMS_eff_t_highpT_2018Down") and isinstance(up, WeightVariation)
    assert up.regions == down.regions == ("nominal",) and up.applies_to == ("mc",) and set(up.replace_weights) == {"tau_id"}
    carriers = {p.name for p in mt.processes if select(mt, p, mt.region("nominal"), up) is not None}
    assert carriers == {p.name for p in mt.processes if mt.kind_of(p) == "mc" and p.role != "data"} and EMBEDDED not in carriers
    # rows: below the threshold; above it with the pT-binned set above, below and around the DM-binned SF; a fake above it
    frame = pd.DataFrame({
        "pt_2": [100.0, 140.0, 150.0, 150.0, 150.0, 150.0, 300.0],
        "gen_match_2": [5, 5, 5, 5, 5, 5, 6],
        "id_wgt_tau_vsJet_Medium_2": [0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 1.00],
        "id_wgt_tau_vsJet_Medium_highpt_2": [0.80, 0.80, 0.95, 0.80, 0.92, 0.88, 0.50],
        "id_wgt_tau_vsJet_Medium_highpt_up_2": [0.85, 0.85, 1.00, 0.84, 0.97, 0.93, 0.60],
        "id_wgt_tau_vsJet_Medium_highpt_down_2": [0.75, 0.75, 0.90, 0.76, 0.87, 0.83, 0.40],
    }).astype({"gen_match_2": np.int32})
    frame = frame.assign(**{name: np.ones(len(frame)) for name in ("puweight", "id_wgt_mu_1", "iso_wgt_mu_1", "id_wgt_tau_vsMu_Tight_2", "id_wgt_tau_vsEle_VVLoose_2", "trg_wgt_single_mu24",
                                                                 "prefiring_wgt", "btag_weight_upart", "topPtReweightWeight")})
    sf_dm = 0.90
    np.testing.assert_allclose(_factor(mt, "TTL", up.name, frame), [1.0, 1.0, 1.00 / sf_dm, 1.0, 0.97 / sf_dm, 0.93 / sf_dm, 1.0])  # max(SF_pt_up, SF_pt, SF_dm) / SF_dm
    np.testing.assert_allclose(_factor(mt, "TTL", down.name, frame), [1.0, 1.0, 1.0, 0.76 / sf_dm, 0.87 / sf_dm, 0.83 / sf_dm, 1.0])  # min(SF_pt_down, SF_pt, SF_dm) / SF_dm
    assert (_factor(mt, "TTL", up.name, frame) >= 1.0).all() and (_factor(mt, "TTL", down.name, frame) <= 1.0).all()  # the envelope holds the nominal
    # tt: the product of both legs, each on its own
    tt = analysis.channel("tt")
    # both legs genuine above the threshold; the second a fake above it; both genuine below it
    two = pd.DataFrame({"pt_1": [200.0, 200.0, 60.0], "pt_2": [150.0, 200.0, 60.0], "gen_match_1": [5, 5, 5], "gen_match_2": [5, 6, 5]}).astype({"gen_match_1": np.int32, "gen_match_2": np.int32})
    for leg in TAU_LEGS["tt"]:
        two[f"id_wgt_tau_vsJet_Medium_{leg}"], two[f"id_wgt_tau_vsJet_Medium_highpt_{leg}"] = 0.9, 0.99
        two[f"id_wgt_tau_vsJet_Medium_highpt_up_{leg}"], two[f"id_wgt_tau_vsJet_Medium_highpt_down_{leg}"] = 1.08, 0.81
    variation = next(v for v in tt.variations if v.name == "CMS_eff_t_highpT_2018Up")
    np.testing.assert_allclose(product(two, list(variation.replace_weights.values())), [1.08 * 1.08, 1.08, 0.9 * 0.9])
    variation = next(v for v in tt.variations if v.name == "CMS_eff_t_highpT_2018Down")
    np.testing.assert_allclose(product(two, list(variation.replace_weights.values())), [0.81 * 0.81, 0.81, 0.9 * 0.9])
    # it comes with the shift model: not in a run without it, not in em, and the CROWN names are those of crown_names
    assert not [v for v in v8(run2, "2018", crown_shifts=False).channel("mt").variations if v.name.startswith(TAU_HIGHPT)]
    assert not [v for v in v8(run2, "2018").channel("em").variations if v.name.startswith(TAU_HIGHPT)]
    assert crown.highpt_columns("Medium", 2) == {"nominal": "id_wgt_tau_vsJet_Medium_highpt_2", "up": "id_wgt_tau_vsJet_Medium_highpt_up_2", "down": "id_wgt_tau_vsJet_Medium_highpt_down_2"}
    assert crown.HIGHPT_THRESHOLD == 140.0


# ---- the parton shower of ttbar (Q6, P8) ----


@pytest.mark.parametrize("era", ERAS)
def test_the_parton_shower_of_ttbar_is_normalised_before_the_acceptance(run2, era):
    analysis = v8(run2, era, embedding=["et", "mt"], ttbar_parton_shower=True)
    assert TTBAR_THEORY == {"ps_isr_ttbar": "ps_isr", "ps_fsr_ttbar": "ps_fsr"}
    for name, channel in analysis.channels.items():
        variations = {v.name: v for v in channel.variations if v.name.startswith(("ps_isr_ttbar", "ps_fsr_ttbar"))}
        assert set(variations) == {f"ps_{shower}_ttbar{d}" for shower in ("isr", "fsr") for d in DIRECTIONS}  # one name for every era: correlated
        ttbar = {p.name for p in channel.processes if p.group == "TT"}
        assert ttbar == ({"TT"} if name == "em" else {"TTT", "TTL"}) and {p.name for p in channel.processes if "ttbar_theory" in p.selection.weights} == ttbar
        assert all({p.name for p in channel.processes if select(channel, p, channel.region("nominal"), v) is not None} == ttbar for v in variations.values())
        samples = channel.samples_of("TT")
        norm = sum(s.xsec * RUNS_NORM[s.nick]["ps_fsrUp"] for s in samples) / sum(s.xsec for s in samples)
        assert len(samples) == 3 and variations["ps_fsr_ttbarUp"].replace_weights == {"ttbar_theory": f"(ps_weight__ps_fsrUp / {round(norm, 6)!r})"}
        assert abs(norm - 1) < 5e-4 and variations["ps_isr_ttbarDown"].regions == ("nominal",)
        frame = pd.DataFrame({"ps_weight__ps_isrDown": [0.9, 1.3]})
        isr = sum(s.xsec * RUNS_NORM[s.nick]["ps_isrDown"] for s in samples) / sum(s.xsec for s in samples)
        np.testing.assert_allclose(product(frame, list(variations["ps_isr_ttbarDown"].replace_weights.values())), np.array([0.9, 1.3]) / round(isr, 6))
    off = v8(run2, era, embedding=["et", "mt"])
    assert not [v for channel in off.channels.values() for v in channel.variations if "ttbar" in v.name and v.name.startswith("ps_")]
    assert not [p for channel in off.channels.values() for p in channel.processes if "ttbar_theory" in p.selection.weights]


# ---- the switches (SH17) ----


@pytest.mark.parametrize("switches,match", [
    ({"jer_regions": True}, "jer_regions requires crown_shifts: true and shift_set: v8"),
    ({"jer_regions": True, "shift_set": "v8", "crown_shifts": False}, "jer_regions requires crown_shifts: true and shift_set: v8"),
    ({"ttbar_parton_shower": True, "crown_shifts": True}, "ttbar_parton_shower requires crown_shifts: true and shift_set: v8"),
    ({"pseudo_shifts": True, "nn_validation": True, **V8}, "pseudo_shifts stand in for the kinematic shifts"),
    ({"shift_set": "v9"}, "shift_set"),
    ({"study_uncertainties": True, "jet_fakes": "mc", **V8}, "study_uncertainties requires jet_fakes: ff"),
])
def test_the_v8_switches_need_the_v8_shifts(run2, switches, match):
    with pytest.raises(AnalysisError, match=match):
        tau(run2, "2018", **switches)


@pytest.mark.parametrize("era", ERAS)
def test_the_study_stand_ins_vanish_with_their_shifts_and_the_other_lnn_stay(run2, era):
    """crown_shifts and study_uncertainties together (SH17): a shift replaces its stand-in; the lnN without a shift
    stay, with their values."""
    base = {"embedding": ["et", "mt"], "study_uncertainties": True, "dy_correction": "njets_btag", "fit_categories": "control"}
    study = tau(run2, era, **base)  # v7b: nominal only
    both = tau(run2, era, **base, **V8)
    old, new = ({(e.name, e.processes, e.value, e.channels) for e in analysis.lnn} for analysis in (study, both))
    gone = {entry[0] for entry in old - new}
    stand_ins = ("CMS_scale_t_emb_dm", "CMS_scale_t_dm", "CMS_eff_t_vsJet", "CMS_eff_t_trigger_ditau_dm")
    assert gone and all(name.startswith(stand_ins) or name == "CMS_eff_e_reco" for name in gone)
    assert {name for name, *_ in old if name.startswith(stand_ins)} == {name for name in gone if name != "CMS_eff_e_reco"}  # every stand-in
    assert {entry[0] for entry in new - old} == {"CMS_eff_e_reco_13TeV"}  # SH27: the name of the CMS list, the same entries
    assert {entry[1:] for entry in old if entry[0] == "CMS_eff_e_reco"} == {entry[1:] for entry in new if entry[0] == "CMS_eff_e_reco_13TeV"}
    stay = {name for name, *_ in new}
    assert f"CMS_eff_m_stat_{era}" in stay and {"eff_e", "eff_m", f"CMS_eff_m_trigger_{era}", f"CMS_eff_e_trigger_{era}", "htt_tjXsec", "QCDscale_HH", f"CMS_htt_dy_table_{era}"} <= stay
    assert (f"CMS_htt_met_mt_acceptance_{era}" in stay) == (era not in RECOIL_ERAS) and ("CMS_htt_hipm_eff_m_2016preVFP" in stay) == (era == "2016preVFP")
    assert (f"CMS_ff_2016half_mt_{era}" in stay) == era.startswith("2016")
    for name, channel in both.channels.items():
        derived = [v.name for v in channel.variations if isinstance(v, ColumnVariation) and v.derived]
        assert [v for v in derived if "dm_nonclosure" in v] == ([] if name in ("et", "em") else [f"CMS_ff_dm_nonclosure_{name}_{era[:4]}Up", f"CMS_ff_dm_nonclosure_{name}_{era[:4]}Down"])
        assert not [v for v in derived if "emb" in v]  # the weight-factor stand-in of the embedding vsJet ID is gone
    # the shifts of the productions up to v6 (embedding only) replace the embedding stand-ins alone
    legacy = tau(run2, era, **base, crown_shifts=True)
    kept = {e.name for e in legacy.lnn}
    assert not [name for name in kept if name.startswith("CMS_scale_t_emb_dm")] and f"CMS_scale_t_dm0_{era}" in kept and "CMS_eff_t_vsJet_syst_alleras" in kept
    assert "CMS_eff_e_reco" in kept and "CMS_eff_e_reco_13TeV" not in kept
    assert not [v for v in legacy.channel("mt").variations if isinstance(v, ColumnVariation) and v.derived and "emb" in v.name]
    assert [v for v in legacy.channel("mt").variations if v.name == f"CMS_eff_t_emb_dm0_vsEleVVLoose_{era}Up" and v.suffix]


@pytest.mark.parametrize("era", ERAS)
def test_the_flat_vsjet_lnn_gives_way_to_the_shapes(run2, era):
    """Without the study switch the flat eff_t_vsJet (2 % per tau_h) stands for the vsJet SF; with the v8 shifts its
    POG components are shapes."""
    flat = f"eff_t_vsJet_$CHANNEL_{era}"
    assert [e for e in tau(run2, era).lnn if e.name == flat] and [e for e in v8(run2, era, crown_shifts=False).lnn if e.name == flat]
    with_shapes = v8(run2, era)
    assert not [e for e in with_shapes.lnn if e.name == flat] and "CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_alleras" in nuisances(with_shapes.channel("mt"))
    assert [e for e in with_shapes.lnn if e.name.startswith("eff_t_vsLep")]  # S10
    without_shapes = v8(run2, era, shape_systematics=False)  # no shape, so the lnN stays
    assert [e for e in without_shapes.lnn if e.name == flat] and not any(isinstance(v, ColumnVariation) for c in without_shapes.channels.values() for v in c.variations)


def test_the_names_of_the_crown_columns_are_in_one_place():
    """What the analysis reads from the CROWN package after the inventory (names fixed there): one module holds them."""
    assert crown.PARTON_SHOWER == ("ps_isr", "ps_fsr") and crown.QCD_SCALES == ("QCDscale_ren", "QCDscale_fac")
    assert crown.pdf_weight(1) == "lhe_pdf_weight_1" and crown.pdf_weight(100) == "lhe_pdf_weight_100" and crown.PDF_MEMBERS == tuple(range(1, 101))
    assert crown.shifted(crown.PS_WEIGHT, "ps_isr", "Up") == "ps_weight__ps_isrUp" and crown.suffix("QCDscale_ren", "Down") == "__QCDscale_renDown"
    assert crown.JER_REGIONS == ("eta0to1p93", "eta1p93to2p5", "eta2p5to3_ptlt50", "eta2p5to3_ptgt50", "eta3to5_ptlt50", "eta3to5_ptgt50")
    assert crown.MUON_ES_REGIONS == {"mt": ("etalt1p2", "eta1p2to2p1"), "em": ("etalt1p2", "eta1p2to2p1", "eta2p1to2p4")}
    sources = [path.read_text() for path in (REPO / "src" / "bbtautau_shapesmith").glob("*.py") if path.name not in ("crown_names.py", "theory_norms.py", "tau_id_systematics.py")]
    for literal in ("lhe_pdf_weight", "ps_weight", "highpt_up", "highpt_down", "genElectron_{region}_{era}", "VSjet_dm_stat", "CMS_res_j_eta", "unclustered_energy_{"):
        assert not [literal for source in sources if literal in source.replace("lhe_pdf_weight_1..100", "")], literal
