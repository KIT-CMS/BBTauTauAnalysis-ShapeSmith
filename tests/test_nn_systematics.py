"""The systematics stages of the NN phase (nn_limit_spec.md 8, 3.6): S0+ the top-pT shape of ttbar (switch ttbar_shape,
SH7) and the heavy-flavour normalisation at >= 3 b tags (heavy_flavour, SH8), S1 the signal theory shapes normalised to
the inclusive rate (signal_theory, SH14; v8 spec SH20, V27). Off by default: a run without them is unchanged."""
import numpy as np
import pandas as pd
import pytest
from shapesmith.events import select
from shapesmith.expressions import product
from shapesmith.histogram import is_part, part_of
from shapesmith.model import AnalysisError, QuadratureSum, Sample, WeightVariation
from shapesmith.skim import needed_columns
from shapesmith.validate import validate

from bbtautau_shapesmith import crown_names as crown
from bbtautau_shapesmith.analysis import build
from bbtautau_shapesmith.constants import ERAS, NN_COLUMNS, TAU_CHANNELS, TAU_LEGS
from bbtautau_shapesmith.processes import EMBEDDED, SIGNAL
from bbtautau_shapesmith.systematics import (
    HEAVY_FLAVOUR, HEAVY_FLAVOUR_LNN, SIGNAL_THEORY, TTBAR_SHAPE, heavy_flavour_variations, signal_theory_sums, signal_theory_variations, theory_norm, ttbar_shape_variations,
)
from bbtautau_shapesmith.theory_norms import RUNS_NORM
from tests.helpers import EMBEDDING_LIST, branches, config, embedding_branches
from tests.run2 import POWHEG, nick
from tests.test_analysis import COMBINATIONS

CHANNELS = ("et", "mt", "tt", "em")
V8 = {"crown_shifts": True, "shift_set": "v8"}  # the theory weights come with the shift production
STAGES = {"ttbar_shape": True, "heavy_flavour": True, "signal_theory": ["qcd_scale", "parton_shower", "pdf"]}
STAGE_NAMES = (TTBAR_SHAPE, HEAVY_FLAVOUR, *(name for names in SIGNAL_THEORY.values() for name in names))
STAGE_WEIGHTS = ("heavy_flavour", "theory")
THEORY_SHIFTS = ("QCDscale_ren", "QCDscale_fac", "ps_isr", "ps_fsr")
# the columns of the shift production that the nominal v7 fixtures lack and the signal reads with SH14: the CROWN shifts
# of lhe_scale_weight and of the parton-shower weight, and the 100 eigenvector weights of the PDF
P6_COLUMNS = {f"lhe_scale_weight__QCDscale_{scale}{d}" for scale in ("ren", "fac") for d in ("Up", "Down")}
P6_COLUMNS |= {f"ps_weight__ps_{shower}{d}" for shower in ("isr", "fsr") for d in ("Up", "Down")} | {f"lhe_pdf_weight_{member}" for member in range(1, 101)}
NODE_SM_2018 = "GluGluToHHTo2B2Tau_TuneCP5_PSWeights_node_SM_13TeV-madgraph-pythia8_RunIISummer20UL18NanoAODv15-150X_mc2018_realistic_v1-v1"  # the signal of the v3 list


def analysis(jet_fakes="ff", embedding=("et", "mt"), **switches):
    embedding = list(embedding) if isinstance(embedding, tuple) else embedding
    sample_lists = ["sm2018_binned_v3", EMBEDDING_LIST] if embedding else ["sm2018_binned_v3"]
    return build(config(CHANNELS, jet_fakes=jet_fakes, embedding=embedding, sample_lists=sample_lists, **switches))


def stage_variations(channel) -> list[str]:
    """The Up/Down variations of the stages (without the eigenvector templates of the PDF, the parts of its sum)."""
    return [v.name for v in channel.variations if v.name.removesuffix("Up").removesuffix("Down") in STAGE_NAMES]


def applies_to(channel, name: str) -> set[str]:
    variation = next(v for v in channel.variations if v.name == name)
    return {p.name for p in channel.processes if select(channel, p, channel.region("nominal"), variation) is not None}


def factor(channel, process: str, name: str, frame: pd.DataFrame) -> np.ndarray:
    """The weight of `process` under the variation over its nominal weight, on `frame`: the weights it replaces, varied
    over nominal; select() replaces them in place and keeps every other weight."""
    variation = next(v for v in channel.variations if v.name == name)
    definition = channel.process(process)
    nominal, varied = (select(channel, definition, channel.region("nominal"), v)[1] for v in (None, variation))
    replaced = [definition.selection.weights[weight] for weight in variation.replace_weights]
    assert varied == [variation.replace_weights.get(weight, expr) for weight, expr in definition.selection.weights.items()] and len(varied) == len(nominal)
    return product(frame, list(variation.replace_weights.values())) / product(frame, replaced)


def weight_frame(**columns) -> pd.DataFrame:
    return pd.DataFrame({name: np.asarray(values, dtype=float) for name, values in columns.items()})


@pytest.mark.parametrize("jet_fakes,embedding", COMBINATIONS)
def test_the_stages_are_off_by_default(jet_fakes, embedding):
    """Without the switches a run is that of before: no stage variation and no added weight (the explicit off is the default)."""
    default = analysis(jet_fakes, embedding)
    assert default == analysis(jet_fakes, embedding, ttbar_shape=False, heavy_flavour=False, signal_theory=[])
    for channel in default.channels.values():
        assert not stage_variations(channel)
        assert not [p.name for p in channel.processes if set(STAGE_WEIGHTS) & set(p.selection.weights)]
        assert sum(isinstance(v, WeightVariation) for v in channel.variations) == 10  # the four b-tag components and prefiring


@pytest.mark.parametrize("jet_fakes,embedding", COMBINATIONS)
def test_the_ttbar_shape_varies_the_processes_with_the_top_pt_weight(jet_fakes, embedding):
    on = analysis(jet_fakes, embedding, ttbar_shape=True)
    off = analysis(jet_fakes, embedding)
    for name, channel in on.channels.items():
        assert stage_variations(channel) == [f"{TTBAR_SHAPE}Up", f"{TTBAR_SHAPE}Down"]
        top = {p.name for p in channel.processes if "top_pt" in p.selection.weights}
        assert top == ({"TT"} if name == "em" else {"TTT", "TTL", "TTJ"} & {p.name for p in channel.processes}) and top
        assert applies_to(channel, f"{TTBAR_SHAPE}Up") == applies_to(channel, f"{TTBAR_SHAPE}Down") == top
        assert channel.processes == off.channel(name).processes  # no weight added: it replaces the top-pT weight
    assert ttbar_shape_variations("2017") == ttbar_shape_variations("2018")  # one nuisance for every era and channel
    (up, down) = ttbar_shape_variations("2018")
    assert (up.regions, up.replace_weights, down.replace_weights) == (("nominal",), {"top_pt": "(2.0 * topPtReweightWeight - 1.0)"}, {"top_pt": "1.0"})


@pytest.mark.parametrize("channel", ["mt", "em"])
def test_the_ttbar_shape_is_with_against_without_the_top_pt_weight(channel):
    """K9: Down without the weight, Up its mirror 2 w - 1 around the nominal (not w^2)."""
    definition = analysis(ttbar_shape=True).channel(channel)
    process = "TT" if channel == "em" else "TTL"
    w = np.array([0.828, 0.95, 1.0, 1.063])  # the range of the v7b skims
    frame = weight_frame(topPtReweightWeight=w)
    np.testing.assert_allclose(factor(definition, process, f"{TTBAR_SHAPE}Down", frame), 1 / w)
    np.testing.assert_allclose(factor(definition, process, f"{TTBAR_SHAPE}Up", frame), (2 * w - 1) / w)


@pytest.mark.parametrize("jet_fakes,embedding", COMBINATIONS)
def test_the_heavy_flavour_component_is_on_the_simulated_backgrounds(jet_fakes, embedding):
    on = analysis(jet_fakes, embedding, heavy_flavour=True)
    for name, channel in on.channels.items():
        assert stage_variations(channel) == [f"{HEAVY_FLAVOUR}Up", f"{HEAVY_FLAVOUR}Down"]
        simulated = {p.name for p in channel.processes if p.role == "background" and p.group != EMBEDDED}
        assert applies_to(channel, f"{HEAVY_FLAVOUR}Up") == applies_to(channel, f"{HEAVY_FLAVOUR}Down") == simulated
        assert {p.name for p in channel.processes if "heavy_flavour" in p.selection.weights} == simulated
        assert all(p.selection.weights["heavy_flavour"] == "1.0" for p in channel.processes if p.name in simulated)  # the nominal stays
        assert not {SIGNAL, EMBEDDED, "data"} & simulated and not {p.name for p in channel.processes if p.role == "auxiliary"} & simulated
        groups = {channel.process(name).group for name in simulated}
        for sample in channel.samples:  # the generator flavour of the b pair: the simulated backgrounds only
            assert ({"bpair_hadronFlavour_1", "bpair_hadronFlavour_2"} <= needed_columns(channel, sample)) == (sample.group in groups), sample.nick
            assert sample.group not in groups or sample.kind == "mc"
    for channel in analysis(jet_fakes, embedding).channels.values():
        assert not any("bpair_hadronFlavour_1" in needed_columns(channel, sample) for sample in channel.samples)


def test_the_heavy_flavour_component_has_three_b_tags_and_a_heavy_b_pair():
    """30 % on the events with >= 3 b tags whose b pair is two b or c jets, Down 1 / 1.30; every other event unchanged."""
    channel = analysis(heavy_flavour=True).channel("mt")
    n_bjets = np.array([2, 3, 3, 3, 3, 4, 3, 1])
    first = np.array([5, 5, 5, 4, 0, 4, 5, 5])
    second = np.array([5, 5, 0, 5, 0, 4, 4, 5])
    component = np.array([False, True, False, True, False, True, True, False])
    frame = weight_frame(n_bjets=n_bjets, bpair_hadronFlavour_1=first, bpair_hadronFlavour_2=second)
    np.testing.assert_allclose(factor(channel, "TTL", f"{HEAVY_FLAVOUR}Up", frame), np.where(component, HEAVY_FLAVOUR_LNN, 1.0))
    np.testing.assert_allclose(factor(channel, "TTL", f"{HEAVY_FLAVOUR}Down", frame), np.where(component, 1 / HEAVY_FLAVOUR_LNN, 1.0), rtol=1e-4)
    assert HEAVY_FLAVOUR_LNN == 1.30 and heavy_flavour_variations("2016preVFP") == heavy_flavour_variations("2018")


def test_the_signal_theory_shapes_are_on_the_signal_normalised_to_the_inclusive_rate():
    on = analysis(**V8, signal_theory=["qcd_scale", "parton_shower", "pdf"])
    names = [f"{name}{d}" for name in ("QCDscale_ren_HH_acc", "QCDscale_fac_HH_acc", "ps_isr_HH", "ps_fsr_HH") for d in ("Up", "Down")]
    norm = RUNS_NORM[NODE_SM_2018]  # of the run's signal sample
    for channel in on.channels.values():
        assert stage_variations(channel) == names
        assert all(applies_to(channel, name) == {SIGNAL} for name in names)
        assert {p.name for p in channel.processes if "theory" in p.selection.weights} == {SIGNAL}
    mt = on.channel("mt")
    up = next(v for v in mt.variations if v.name == "QCDscale_ren_HH_accUp")
    assert up.replace_weights == {"theory": f"(lhe_scale_weight__QCDscale_renUp / {norm['QCDscale_renUp']!r})"} and up.regions == ("nominal",)
    assert next(v for v in mt.variations if v.name == "ps_fsr_HHDown").replace_weights == {"theory": f"(ps_weight__ps_fsrDown / {norm['ps_fsrDown']!r})"}
    # an event whose varied weight is the inclusive mean keeps its weight: the inclusive rate stays (no double counting
    # with QCDscale_HH)
    columns = {crown.shifted(crown.LHE_SCALE_WEIGHT if shift.startswith("QCD") else crown.PS_WEIGHT, shift, d): norm[f"{shift}{d}"] for shift in THEORY_SHIFTS for d in ("Up", "Down")}
    frame = weight_frame(**{column: np.full(2, value) for column, value in columns.items()})
    for name in names:
        np.testing.assert_allclose(factor(mt, SIGNAL, name, frame), 1.0)
    frame = weight_frame(**{column: np.array([0.8, 1.2]) for column in columns})
    np.testing.assert_allclose(factor(mt, SIGNAL, "QCDscale_fac_HH_accDown", frame), np.array([0.8, 1.2]) / norm["QCDscale_facDown"])
    only_scales = analysis(**V8, signal_theory=["qcd_scale"]).channel("mt")
    assert stage_variations(only_scales) == names[:4] and not [e for e in only_scales.estimators if isinstance(e, QuadratureSum)]


def test_the_pdf_of_the_signal_is_the_quadrature_sum_of_its_eigenvector_templates():
    """V27: one template per Hessian eigenvector, each weight over its inclusive normalisation, summed in quadrature
    per bin by the estimator; no weight of the sum per event."""
    on = analysis(**V8, signal_theory=["pdf"])
    validate(on)
    norms = RUNS_NORM[NODE_SM_2018]["pdf"]
    for channel in on.channels.values():
        parts = [v for v in channel.variations if is_part(v.name) and v.name.startswith("pdf_HH_acc")]
        assert [v.name for v in parts] == [part_of("pdf_HH_acc", f"{member}") for member in range(1, 101)]
        assert all(isinstance(v, WeightVariation) and v.regions == ("nominal",) and applies_to(channel, v.name) == {SIGNAL} for v in parts)
        assert parts[0].replace_weights == {"theory": f"(lhe_pdf_weight_1 / {norms[0]!r})"} and parts[99].replace_weights == {"theory": f"(lhe_pdf_weight_100 / {norms[99]!r})"}
        assert [e for e in channel.estimators if isinstance(e, QuadratureSum)] == [QuadratureSum("pdf_HH_acc", tuple(f"{member}" for member in range(1, 101)))]
        assert not stage_variations(channel)  # no Up/Down weight: the sum makes them
    assert signal_theory_sums("2017", ["qcd_scale", "parton_shower"]) == () and signal_theory_sums("2017", ["pdf"]) == signal_theory_sums("2018", ["pdf"])  # one nuisance for every era


@pytest.mark.parametrize("switches,match", [
    ({"crown_shifts": False, "shift_set": "v8", "signal_theory": ["qcd_scale"]}, "signal_theory requires crown_shifts: true and shift_set: v8"),
    ({"crown_shifts": True, "signal_theory": ["qcd_scale"]}, "signal_theory requires crown_shifts: true and shift_set: v8"),  # the shifts of the productions up to v6
    ({**V8, "signal_theory": ["qcd_scale"], "signal_sample": "node_sm"}, "signal_theory requires signal_sample: powheg"),
    ({**V8, "signal_theory": ["pdf", "pdf"]}, "signal_theory lists a kind twice"),
    ({**V8, "signal_theory": ["scale"]}, "signal_theory"),
])
def test_the_signal_theory_needs_the_shift_production_and_the_powheg_signal(switches, match):
    with pytest.raises(AnalysisError, match=match):
        analysis(**switches)


def test_the_signal_theory_normalisation_of_every_era():
    """Every varied weight of the signal has its inclusive normalisation in every era (theory_norms.RUNS_NORM, from the
    NanoAOD Runs sums), for the powheg kappa_lambda = 1 sample and the other kappa_lambda points (Q8); a sample without
    one fails the build."""
    keys = {f"{shift}{d}" for shift in THEORY_SHIFTS for d in ("Up", "Down")}
    def powheg(kappa_lambda: str, era: str) -> dict:
        (name,) = [n for n, entry in RUNS_NORM.items() if n.startswith(f"GluGluToHHTo2B2T_kl-{kappa_lambda}_kt-1p00_c2-0p00_") and entry["era"] == era]
        return RUNS_NORM[name]

    for era in ERAS:
        for kappa_lambda in ("0p00", "1p00", "2p45", "5p00"):
            norm = powheg(kappa_lambda, era)
            assert keys | {"pdf", "era", "sample_type"} == set(norm) and norm["sample_type"] == "hh2b2tau"
            assert all(abs(norm[key] - 1) < 0.002 for key in keys if key.startswith("ps_"))
            assert len(norm["pdf"]) == 100 and all(0.99 < value < 1.01 for value in norm["pdf"])
            assert 0.0090 < np.sqrt(sum((value - 1) ** 2 for value in norm["pdf"])) < 0.0105  # the inclusive Hessian PDF uncertainty, 1 %
        norm = RUNS_NORM[nick(POWHEG["1p00"], era)]  # kappa_lambda = 1: muR -10.6 / +12.0 %, muF -1.8 / +2.3 %
        assert 0.89 < norm["QCDscale_renUp"] < 0.90 and 1.11 < norm["QCDscale_renDown"] < 1.13
        assert 0.98 < norm["QCDscale_facUp"] < 0.99 and 1.02 < norm["QCDscale_facDown"] < 1.03
        assert norm == powheg("1p00", era)
        assert all(abs(powheg(other, era)["QCDscale_renUp"] - norm["QCDscale_renUp"]) > 0.005 for other in ("0p00", "2p45", "5p00"))  # their own: 0.886, 0.888, 0.864 against 0.894
        signal = (Sample(nick(POWHEG["1p00"], era), SIGNAL, "mc", xsec=0.002221),)
        variations = signal_theory_variations(era, SIGNAL_THEORY, signal)
        assert len(variations) == 8 + 100 and theory_norm(signal, "QCDscale_renUp") == norm["QCDscale_renUp"]
    with pytest.raises(AnalysisError, match="no inclusive normalisation of QCDscale_renUp for SomeOtherSignal"):
        signal_theory_variations("2018", ["qcd_scale"], (Sample("SomeOtherSignal", SIGNAL, "mc"),))
    with pytest.raises(AnalysisError, match="no inclusive normalisation of pdf for a process without samples"):
        signal_theory_variations("2018", ["pdf"], ())


def test_the_normalisation_of_several_samples_is_their_cross_section_weighted_mean():
    """ttbar is three samples: one normalisation keeps their summed inclusive rate."""
    names = sorted(n for n, entry in RUNS_NORM.items() if entry["era"] == "2018" and entry["sample_type"] == "ttbar" and "_TuneCP5_13TeV" in n and "hdamp" not in n)
    assert [n.split("_")[0] for n in names] == ["TTTo2L2Nu", "TTToHadronic", "TTToSemiLeptonic"]
    xsecs = (88.29, 377.96, 365.34)
    samples = tuple(Sample(name, "TT", "mc", xsec=xsec) for name, xsec in zip(names, xsecs))
    expected = sum(xsec * RUNS_NORM[name]["ps_fsrDown"] for name, xsec in zip(names, xsecs)) / sum(xsecs)
    assert theory_norm(samples, "ps_fsrDown") == pytest.approx(expected, abs=1e-6)
    assert all(abs(RUNS_NORM[name][f"ps_{shower}{d}"] - 1) < 5e-4 for name in names for shower in ("isr", "fsr") for d in ("Up", "Down"))  # W0: 1 within 4e-4


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_the_stage_columns_exist(channel):
    """S0+ reads n-tuple columns of the v7b production (the top-pT weight, the b-pair flavour of the re-skim); S1 reads
    the columns of the shift production only: the theory weights on the signal and, with the shift model of v8, the
    pT-binned vsJet SF of every simulated sample (V35)."""
    on = analysis(**V8, nn_friend=True, fit_categories="nn", **STAGES)
    definition = on.channel(channel)
    friends = NN_COLUMNS | {"fake_factor", "fake_factor_1", "fake_factor_2"}
    highpt = {column for leg in TAU_LEGS[channel] for column in crown.highpt_columns("Medium", leg).values()}
    for sample in definition.samples:
        available = embedding_branches(channel) if sample.kind == "embedding" else branches(channel) | {"npartons"}
        missing = needed_columns(definition, sample) - available - friends
        expected = (highpt if sample.kind == "mc" else set()) | (P6_COLUMNS if sample.group == SIGNAL else set())
        assert missing == expected, (sample.nick, missing ^ expected)
