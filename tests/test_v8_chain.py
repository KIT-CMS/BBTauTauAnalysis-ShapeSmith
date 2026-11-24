"""The v8 shift model from n-tuples to datacards on synthetic files (no v8 production exists yet): mt of 2018 with one
sample each of the data, the signal, the embedding, DY and ttbar (the other groups left out of the process table: the
chain evaluates every cut and weight per sample and shift, some hundred thousand expressions), each with every shift of
the inventory in its main tree and the friends NN, FastMTT and fake factors; skim -> hist -> estimate -> datacards with
every switch of S1 on. The numbers of the synthetic files are chosen so that every result is known: constant weights,
and a tau pT that every shift moves across the 30 GeV cut."""
import dataclasses

import numpy as np
import pytest
from shapesmith.config import FriendConfig, NtupleConfig, RunConfig
from shapesmith.datacards import write_datacard
from shapesmith.estimates import run_estimates
from shapesmith.fill import run_hist
from shapesmith.histogram import NOMINAL_VARIATION, HistKey, part_of
from shapesmith.model import DataMinus, declared_only
from shapesmith.skim import column_variations, needed_columns, run_skim
from shapesmith.store import read_skims
from shapesmith.testing import make_ntuple
from shapesmith.validate import validate

from bbtautau_shapesmith import crown_names as crown
from bbtautau_shapesmith.analysis import build
from bbtautau_shapesmith.processes import EMBEDDED, SIGNAL
from bbtautau_shapesmith.systematics import theory_norm
from tests.helpers import DATABASE, TAU_ANALYSIS
from tests.run2 import nick
from tests.test_run2 import run2  # noqa: F401 (the fixture)
from tests.test_v8_shifts import FF_SUFFIXES, expected_nuisances

ERA = "2018"
N = 60
# one sample of five groups of the mt channel: nick prefixes of the 2018 inventories
PREFIXES = ("SingleMuon_Run2018A", "GluGluToHHTo2B2T_kl-1p00", "DYJetsToLL_M-10to50", "TTToSemiLeptonic")
GROUPS = {"data", SIGNAL, EMBEDDED, "DY", "TT"}
EMBEDDING_PREFIX = "EmbeddingRun2018A_cwinter-embedding_2018UL_mutau"
NN_PREFIX = "nn_m3abnv1_"
SWITCHES = {
    "jet_fakes": "ff", "embedding": ["mt"], "shift_set": "v8", "crown_shifts": True, "study_uncertainties": True, "fit_categories": "control", "nn_friend": True, "nn_prefix": NN_PREFIX,
    "fastmtt_friend": True, "signal_theory": ["qcd_scale", "parton_shower", "pdf"], "ttbar_parton_shower": True, "ttbar_shape": True,
}
INDEX = np.arange(N)
# the tau pT, by the last digit of the event number: above 140 GeV (2, 3), just above (4, 7) and just below (5, 8) the cut
PT_2 = np.select([np.isin(INDEX % 10, (2, 3)), np.isin(INDEX % 10, (4, 7)), np.isin(INDEX % 10, (5, 8))], [150.0, 30.4, 29.8], 45.0)
UP, DOWN = 1.03, 0.97  # of every shifted pt_2: 29.8 passes Up, 30.4 fails Down
JER_REGION = "eta0to1p93"  # the one JER region with the full shift
PDF_SHIFT = {1: 1.03, 2: 0.96}  # the eigenvectors that move the signal, as a factor on their normalisation; the others equal theirs


def _friend(name: str) -> bool:
    return name.startswith((NN_PREFIX, "fake_factor")) or name.endswith("_fastmtt")


def _column(name: str, sample) -> np.ndarray:
    """The nominal value of a column, the same for every sample: every event passes the nominal mt selection but
    those below the tau pT cut; every third is anti-isolated, every sixth same sign; the hadronic tau genuine, the
    light lepton from a tau in every other event (T, else L); every weight 1."""
    floats = {
        "pt_2": PT_2, "pt_1": 40.0, "iso_1": 0.05, "mt_1": 30.0, "bpair_pt_2": 40.0, "fake_factor": 0.2, "topPtReweightWeight": 0.95,
        "q_2": np.where(INDEX % 6 == 0, 1.0, -1.0), "id_tau_vsJet_Medium_2": np.where(INDEX % 3 == 0, 0.0, 1.0),
        "L1PreFiringWeight_Up": 1.02, "L1PreFiringWeight_Dn": 0.98, f"{NN_PREFIX}predicted_max_value": 0.5, "m_fastmtt": 100.0,
        "lhe_scale_weight__QCDscale_renUp": 1.1, "lhe_scale_weight__QCDscale_renDown": 0.9, "lhe_scale_weight__QCDscale_facUp": 1.02, "lhe_scale_weight__QCDscale_facDown": 0.98,
        "ps_weight__ps_isrUp": 1.05, "ps_weight__ps_isrDown": 0.95, "ps_weight__ps_fsrUp": 0.9, "ps_weight__ps_fsrDown": 1.2,
        "id_wgt_tau_vsJet_Medium_highpt_2": 0.95, "id_wgt_tau_vsJet_Medium_highpt_up_2": 1.05, "id_wgt_tau_vsJet_Medium_highpt_down_2": 0.85,
    }
    integers = {"event": INDEX, "run": 1, "lumi": 1, "gen_match_1": np.where(INDEX % 2 == 0, 4, 2), "gen_match_2": 5, "n_bjets": 1 + INDEX % 2, "n_jets": 2 + INDEX % 3,
                "tau_decaymode_2": np.array([0, 1, 10, 11])[INDEX % 4], f"{NN_PREFIX}predicted_class": INDEX % 5, "npartons": 1}
    if name in integers:
        return np.broadcast_to(integers[name], N).astype(np.uint64 if name in ("event", "run", "lumi") else np.int32)
    if name.startswith("lhe_pdf_weight_"):
        member = int(name.rsplit("_", 1)[1])
        return np.full(N, theory_norm((sample,), "pdf")[member - 1] * PDF_SHIFT.get(member, 1.0), dtype=np.float32)
    value = floats.get(name, 0.0 if name.endswith("_veto") else 1.0)
    return np.broadcast_to(value, N).astype(np.float32)


def _write(base, friends, channel, sample) -> None:
    """The main n-tuple and the friend of a sample: its needed columns, pt_2 under every declared CROWN shift (the
    declared-only ones too), the NN and FastMTT columns under each of them, the fake factor under its own shifts."""
    columns = needed_columns(channel, sample)
    main = {name: _column(name, sample) for name in sorted(columns) if not _friend(name)}
    friend = {name: _column(name, sample) for name in sorted(columns) if _friend(name)}
    for variation in (v for v in column_variations(channel, sample) if v.suffix):  # the CROWN shifts, not the derived variations
        factor = UP if variation.suffix.endswith("Up") else DOWN
        if "CMS_res_j_" in variation.suffix and JER_REGION not in variation.suffix:
            factor = 1 + (factor - 1) / 100  # the other five JER regions move no event across the cut
        if variation.suffix in FF_SUFFIXES:
            friend["fake_factor" + variation.suffix] = (friend["fake_factor"] * (1.1 if factor == UP else 0.9)).astype(np.float32)
            continue
        main["pt_2" + variation.suffix] = (main["pt_2"] * factor).astype(np.float32)
        friend.update({name + variation.suffix: values for name, values in friend.items() if "__" not in name and not name.startswith("fake_factor")})
    for directory, tree in ((base, main), (friends, friend)):
        make_ntuple(directory / ERA / sample.nick / channel.name / f"{sample.nick}_0.root", tree, {"era": ERA} if tree is main else None)


@pytest.fixture(scope="module")
def chain(run2, tmp_path_factory):
    """The run on the synthetic n-tuples: (config, analysis, histograms after estimate, the card)."""
    from bbtautau_shapesmith import samples as samples_module

    tmp_path = tmp_path_factory.mktemp("v8_chain")
    nicks = [nick(prefix, ERA) for prefix in PREFIXES] + [nick(EMBEDDING_PREFIX, ERA, "embedding")]
    (samples_module.INVENTORY_DIR / "v8_chain.txt").write_text("\n".join(nicks) + "\n")
    config = RunConfig(
        analysis=TAU_ANALYSIS, era=ERA, channels=["mt"], switches={**SWITCHES, "sample_lists": ["v8_chain"]}, sample_database=DATABASE,
        ntuples=NtupleConfig(base=str(tmp_path / "ntuples"), friends=[FriendConfig(base=str(tmp_path / "friends"))]), skim_dir=tmp_path / "skims", output_dir=tmp_path / "output", workers=1,
    )
    analysis = build(config)
    # the process table of the five groups: the processes of the others (no samples here) out, also of the jet-fake estimate
    channel = analysis.channel("mt")
    assert {s.group for s in channel.samples} == GROUPS
    processes = tuple(p for p in channel.processes if p.group in GROUPS)
    kept = {p.name for p in processes}
    estimators = tuple(dataclasses.replace(e, subtract=tuple(name for name in e.subtract if name in kept)) if isinstance(e, DataMinus) else e for e in channel.estimators)
    channel = dataclasses.replace(channel, processes=processes, estimators=estimators)
    analysis = dataclasses.replace(analysis, channels={"mt": channel})
    validate(analysis)
    for sample in channel.samples:
        _write(tmp_path / "ntuples", tmp_path / "friends", channel, sample)
    run_skim(config, analysis, ["mt"])
    hset = run_hist(config, analysis, ["mt"], False, None, True, None, config.output_dir / "shapes.root")
    run_estimates(hset, analysis, ["mt"])
    card = write_datacard(hset, analysis, ["mt"], config.output_dir / "datacards" / "mt", min_background=None)
    return config, analysis, hset, card


def _yield(hset, process, variation=NOMINAL_VARIATION, region="nominal"):
    return hset[HistKey("mt", "nominal", process, region, variation, "yield")].sum()


def test_the_skims_hold_the_filled_shifts_and_no_declared_only_one(chain):
    config, analysis, _, _ = chain
    channel = analysis.channel("mt")
    assert [p.name for p in channel.processes] == ["data", SIGNAL, EMBEDDED, "ZL", "TTL", "TTT"] and channel.estimators[0] == DataMinus("jetFakes", "anti_iso", (EMBEDDED, "ZL", "TTL"))
    for sample in channel.samples:
        skim = read_skims(config.skim_dir, "mt", [sample.nick], None)
        declared = {v.suffix: v for v in column_variations(channel, sample) if v.suffix}
        shifted = {"__" + name.split("__", 1)[1] for name in skim.columns if "__" in name and not name.startswith(("lhe_scale_weight__", "ps_weight__"))}
        assert shifted == {suffix for suffix, v in declared.items() if not declared_only(v)}, sample.nick
        # the event at 29.8 GeV passes only under an Up shift: kept where the sample has a shift that is read
        kept = set(skim["event"][np.isclose(skim["pt_2"], 29.8)])
        assert (len(kept) > 0) == (sample.kind != "data"), sample.nick
    ttbar = next(s for s in channel.samples if s.group == "TT")
    skim = read_skims(config.skim_dir, "mt", [ttbar.nick], None)
    assert f"{NN_PREFIX}predicted_class__CMS_scale_j_AbsoluteUp" in skim.columns and "m_fastmtt__CMS_res_j_eta0to1p93_2018Down" in skim.columns
    assert "pt_2__CMS_eff_m_id_2018Up" not in skim.columns and "pt_2__CMS_scale_m_etalt1p2_2018Up" not in skim.columns and "pt_2__QCDscale_renUp" not in skim.columns


def test_the_histograms_follow_the_shifts(chain):
    """Per event of a simulated sample with unit weights the nominal region holds the isolated, opposite-sign events
    above 30 GeV; an Up shift adds those at 29.8 GeV, a Down shift drops those at 30.4 GeV."""
    _, analysis, hset, _ = chain
    channel = analysis.channel("mt")
    selected = (INDEX % 3 != 0) & (INDEX % 6 != 0)
    lepton_fake = selected & (INDEX % 2 == 1)  # the L part: the muon not from a tau
    counts = {name: np.sum(lepton_fake & mask) for name, mask in (("nominal", PT_2 > 30), ("up", PT_2 * UP > 30), ("down", PT_2 * DOWN > 30))}
    assert counts["up"] > counts["nominal"] > counts["down"] > 0
    unit = _yield(hset, "TTL") / (counts["nominal"] * 0.95)  # the weight of one ttbar event (the top-pT weight is 0.95)
    for nuisance in ("CMS_scale_j_Absolute", "CMS_scale_j_EC2_2018", "CMS_scale_met_unclustered_energy_2018", "CMS_scale_e_13TeV", "CMS_scale_t_dm10_2018", "CMS_pileup",
                     "CMS_scale_t_DeepTau2018v2p5_DM0_genMuon_2018", "CMS_fake_m_WH2_2018", "CMS_eff_t_DeepTau2018v2p5_VSjet_dm_syst_TES_dm1_2018"):
        assert _yield(hset, "TTL", f"{nuisance}Up") == pytest.approx(counts["up"] * 0.95 * unit) and _yield(hset, "TTL", f"{nuisance}Down") == pytest.approx(counts["down"] * 0.95 * unit)
    # the JER: the sum of its six regions, of which one moves events here
    moved, unmoved = (_yield(hset, "TTL", part_of("CMS_res_j_2018Up", region)) for region in (JER_REGION, "eta3to5_ptlt50"))
    assert moved == pytest.approx(counts["up"] * 0.95 * unit) and unmoved == pytest.approx(_yield(hset, "TTL"))
    assert _yield(hset, "TTL", "CMS_res_j_2018Up") == pytest.approx(moved) and _yield(hset, "TTL", "CMS_res_j_2018Down") == pytest.approx(counts["down"] * 0.95 * unit)
    assert {key.variation for key in hset.select(process="TTL") if key.variation.startswith("CMS_res_j_2018Up%")} == {part_of("CMS_res_j_2018Up", region) for region in crown.JER_REGIONS}
    # no histogram of a declared-only shift, none of a CROWN shift outside the nominal region, no recoil on ttbar
    filled = {key.variation for key in hset}
    declared = {v.name for v in channel.variations if declared_only(v)}
    assert len(declared) == 2 * (3 + 2 + 4) and not filled & declared
    crown_shifts = {v.name for v in channel.variations if v.name not in declared and getattr(v, "suffix", "") and v.suffix not in FF_SUFFIXES}
    assert {key.region for key in hset if key.variation in crown_shifts} == {"nominal"}
    assert {key.process for key in hset if key.variation.startswith("CMS_htt_boson_")} == {"ZL"}  # the recoil of 2018: DY (W is in the jet fakes)
    assert not hset.select(process="ZL", variation="CMS_scale_met_unclustered_energy_2018Up") and hset.select(process="TTL", variation="CMS_scale_met_unclustered_energy_2018Up")
    assert {key.variation for key in hset.select(process="jetFakes") if key.variation != NOMINAL_VARIATION} <= {v.name for v in channel.variations if "CMS_ff_" in v.name}  # V31


def test_the_embedded_sample_shares_half_of_its_tau_es_with_the_simulation(chain):
    _, _, hset, _ = chain
    nominal = _yield(hset, EMBEDDED)
    for token, decay_modes in (("0", (0,)), ("1", (1,)), ("1011", (10, 11))):
        own = f"CMS_scale_t_emb_dm{token}_vsEleVVLoose_2018"
        shifts = {dm: _yield(hset, EMBEDDED, part_of(f"{own}Up", f"dm{dm}")) - nominal for dm in decay_modes}
        assert all(shift > 0 for shift in shifts.values())
        assert _yield(hset, EMBEDDED, f"{own}Up") == pytest.approx(nominal + 0.866 * sum(shifts.values()))
        for dm, shift in shifts.items():
            assert _yield(hset, EMBEDDED, f"CMS_scale_t_dm{dm}_2018Up") == pytest.approx(nominal + 0.5 * shift)
            assert np.hypot(0.866 * shift, 0.5 * shift) == pytest.approx(shift, rel=1e-4)
    assert _yield(hset, EMBEDDED, "CMS_eff_t_emb_dm0_vsEleVVLoose_2018Up") > nominal  # the vsJet ID: its shift whole
    assert not hset.select(process=EMBEDDED, variation="CMS_res_j_2018Up") and not hset.select(process=EMBEDDED, variation="CMS_pileupUp")


def test_the_theory_shapes_keep_the_inclusive_rate_out(chain):
    _, analysis, hset, _ = chain
    channel = analysis.channel("mt")
    signal, ttbar = channel.samples_of(SIGNAL), channel.samples_of("TT")
    nominal = _yield(hset, SIGNAL)
    for name, weight, key in (("QCDscale_ren_HH_accUp", 1.1, "QCDscale_renUp"), ("QCDscale_fac_HH_accDown", 0.98, "QCDscale_facDown"), ("ps_isr_HHDown", 0.95, "ps_isrDown"), ("ps_fsr_HHUp", 0.9, "ps_fsrUp")):
        assert _yield(hset, SIGNAL, name) == pytest.approx(nominal * weight / theory_norm(signal, key), rel=1e-6)
    assert _yield(hset, "TTL", "ps_fsr_ttbarDown") == pytest.approx(_yield(hset, "TTL") * 1.2 / theory_norm(ttbar, "ps_fsrDown"), rel=1e-6)
    assert not hset.select(process="TTL", variation="ps_fsr_HHUp") and not hset.select(process=SIGNAL, variation="ps_fsr_ttbarUp") and not hset.select(process="ZL", variation="ps_fsr_ttbarUp")
    # the PDF: two eigenvectors move the yield by +3 % and -4 %, the other 98 by nothing: 5 % in quadrature
    assert _yield(hset, SIGNAL, part_of("pdf_HH_acc", "1")) == pytest.approx(1.03 * nominal, rel=1e-6) and _yield(hset, SIGNAL, part_of("pdf_HH_acc", "57")) == pytest.approx(nominal, rel=1e-6)
    assert _yield(hset, SIGNAL, "pdf_HH_accUp") == pytest.approx(1.05 * nominal, rel=1e-5) and _yield(hset, SIGNAL, "pdf_HH_accDown") == pytest.approx(0.95 * nominal, rel=1e-5)


def test_the_highpt_shape_moves_the_genuine_taus_above_140_gev(chain):
    """SF_dm 1, the pT-binned set 0.95 (+0.10, -0.10): above 140 GeV Up is max(1.05, 0.95, 1) = 1.05, Down min(0.85, 0.95, 1) = 0.85."""
    _, _, hset, _ = chain
    selected = (INDEX % 3 != 0) & (INDEX % 6 != 0) & (INDEX % 2 == 1) & (PT_2 > 30)
    high = np.sum(selected & (PT_2 > crown.HIGHPT_THRESHOLD)) / np.sum(selected)
    assert 0 < high < 1
    nominal = _yield(hset, "TTL")
    assert _yield(hset, "TTL", "CMS_eff_t_highpT_2018Up") == pytest.approx(nominal * (1 + 0.05 * high)) and _yield(hset, "TTL", "CMS_eff_t_highpT_2018Down") == pytest.approx(nominal * (1 - 0.15 * high))
    assert not hset.select(process=EMBEDDED, variation="CMS_eff_t_highpT_2018Up")


def test_the_card_of_the_chain(chain):
    _, analysis, _, card = chain
    text = card.read_text()
    kinds = {line.split()[0]: line.split()[1] for line in text.splitlines() if " lnN " in line or " shape " in line}
    shapes = {name for name, kind in kinds.items() if kind == "shape"}
    theory = {"QCDscale_ren_HH_acc", "QCDscale_fac_HH_acc", "ps_isr_HH", "ps_fsr_HH", "pdf_HH_acc", "ps_isr_ttbar", "ps_fsr_ttbar", "CMS_htt_ttbarShape"}
    others = {name for name in shapes if name.startswith(("CMS_ff_", "CMS_btag_", "CMS_htt_emb_ttbar"))}
    assert shapes - others == expected_nuisances("mt", ERA, embedded=True) | theory | {"CMS_eff_t_highpT_2018"}
    assert others and kinds["CMS_prefiring"] == kinds["eff_m"] == kinds["CMS_eff_m_trigger_2018"] == "lnN"
    for absent in ("CMS_scale_m_", "CMS_eff_m_id", "CMS_eff_m_iso", "QCDscale_ren ", "ps_isr ", "%", "CMS_eff_t_vsJet", "CMS_eff_e_"):
        assert absent not in text, absent
    assert (card.parent / "common" / f"htt_input_{ERA}.root").exists() and "htt_mt_1_2018" in text
    assert analysis.channel("mt").categories[0].name == "nominal" and "hi" not in text.split("observation")[0]  # one control category: nothing unblinded
