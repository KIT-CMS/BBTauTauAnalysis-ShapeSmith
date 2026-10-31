"""The systematics stages of the NN phase (nn_limit_spec.md 8, 3.6): S0+ the top-pT shape of ttbar (switch ttbar_shape,
SH7) and the heavy-flavour normalisation at >= 3 b tags (heavy_flavour, SH8), S1 the signal theory shapes normalised to
the inclusive rate (signal_theory, SH14). Off by default: a run without them is unchanged."""
import numpy as np
import pandas as pd
import pytest
from shapesmith.events import select
from shapesmith.expressions import product
from shapesmith.model import AnalysisError, WeightVariation
from shapesmith.skim import needed_columns

from bbtautau_shapesmith import systematics
from bbtautau_shapesmith.analysis import build
from bbtautau_shapesmith.constants import ERAS, NN_COLUMNS, TAU_CHANNELS
from bbtautau_shapesmith.processes import EMBEDDED, SIGNAL
from bbtautau_shapesmith.systematics import (
    HEAVY_FLAVOUR, HEAVY_FLAVOUR_LNN, SIGNAL_THEORY, SIGNAL_THEORY_INDICES, SIGNAL_THEORY_NORM, TTBAR_SHAPE, heavy_flavour_variations, signal_theory_variations,
    ttbar_shape_variations,
)
from tests.helpers import EMBEDDING_LIST, branches, config, embedding_branches
from tests.test_analysis import COMBINATIONS

CHANNELS = ("et", "mt", "tt", "em")
STAGES = {"ttbar_shape": True, "heavy_flavour": True, "signal_theory": ["qcd_scale", "parton_shower", "pdf"]}
STAGE_NAMES = (TTBAR_SHAPE, HEAVY_FLAVOUR, *(name for names in SIGNAL_THEORY.values() for name in names))
STAGE_WEIGHTS = ("heavy_flavour", "theory")
# the columns of the shift production P6 that the nominal v7 fixtures lack: the CROWN shifts of lhe_scale_weight and
# the parton-shower and PDF weights of SH14
P6_COLUMNS = {column.format(direction=d) for columns in SIGNAL_THEORY.values() for column in columns.values() for d in ("Up", "Down")}


def analysis(jet_fakes="ff", embedding=("et", "mt"), **switches):
    embedding = list(embedding) if isinstance(embedding, tuple) else embedding
    sample_lists = ["sm2018_binned_v3", EMBEDDING_LIST] if embedding else ["sm2018_binned_v3"]
    return build(config(CHANNELS, jet_fakes=jet_fakes, embedding=embedding, sample_lists=sample_lists, **switches))


def stage_variations(channel) -> list[str]:
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
    on = analysis(crown_shifts=True, signal_theory=["qcd_scale", "parton_shower", "pdf"])
    names = [f"{name}{d}" for kind in ("qcd_scale", "parton_shower", "pdf") for name in SIGNAL_THEORY[kind] for d in ("Up", "Down")]
    assert names == ["QCDscale_ren_HH_accUp", "QCDscale_ren_HH_accDown", "QCDscale_fac_HH_accUp", "QCDscale_fac_HH_accDown",
                     "ps_isr_HHUp", "ps_isr_HHDown", "ps_fsr_HHUp", "ps_fsr_HHDown", "pdf_HH_accUp", "pdf_HH_accDown"]
    for channel in on.channels.values():
        assert stage_variations(channel) == names
        assert all(applies_to(channel, name) == {SIGNAL} for name in names)
        assert {p.name for p in channel.processes if "theory" in p.selection.weights} == {SIGNAL}
    mt = on.channel("mt")
    up = next(v for v in mt.variations if v.name == "QCDscale_ren_HH_accUp")
    assert up.replace_weights == {"theory": "(lhe_scale_weight__QCDscale_renUp / 0.89355)"} and up.regions == ("nominal",)
    # an event whose varied weight is the inclusive mean keeps its weight: the inclusive rate stays (no double counting
    # with QCDscale_HH)
    norm = SIGNAL_THEORY_NORM["2018"]
    frame = weight_frame(**{column: np.full(2, norm[column.split("__")[1]]) for column in P6_COLUMNS})
    for name in names:
        np.testing.assert_allclose(factor(mt, SIGNAL, name, frame), 1.0)
    frame = weight_frame(**{column: np.array([0.8, 1.2]) for column in P6_COLUMNS})
    np.testing.assert_allclose(factor(mt, SIGNAL, "QCDscale_fac_HH_accDown", frame), np.array([0.8, 1.2]) / norm["QCDscale_facDown"])
    only_scales = analysis(crown_shifts=True, signal_theory=["qcd_scale"]).channel("mt")
    assert stage_variations(only_scales) == names[:4]


@pytest.mark.parametrize("switches,match", [
    ({"crown_shifts": False, "signal_theory": ["qcd_scale"]}, "signal_theory requires crown_shifts: true"),
    ({"crown_shifts": True, "signal_theory": ["qcd_scale"], "signal_sample": "node_sm"}, "signal_theory requires signal_sample: powheg"),
    ({"crown_shifts": True, "signal_theory": ["pdf", "pdf"]}, "signal_theory lists a kind twice"),
    ({"crown_shifts": True, "signal_theory": ["scale"]}, "signal_theory"),
])
def test_the_signal_theory_needs_the_shift_production_and_the_powheg_signal(switches, match):
    with pytest.raises(AnalysisError, match=match):
        analysis(**switches)


def test_the_signal_theory_normalisation_of_every_era(monkeypatch):
    """Every varied weight has its inclusive mean in every era (scripts/signal_theory_sums.py: the NanoAOD Runs sums and
    the PDF weights), and the means are those of the scale variations of the powheg sample."""
    columns = {column.format(direction=d).split("__")[1] for names in SIGNAL_THEORY.values() for column in names.values() for d in ("Up", "Down")}
    assert columns == set(SIGNAL_THEORY_INDICES)
    for era in ERAS:
        norm = SIGNAL_THEORY_NORM[era]
        assert set(norm) == columns
        assert 0.89 < norm["QCDscale_renUp"] < 0.90 and 1.11 < norm["QCDscale_renDown"] < 1.13  # muR: -10.6 / +12.0 %
        assert 0.98 < norm["QCDscale_facUp"] < 0.99 and 1.02 < norm["QCDscale_facDown"] < 1.03
        assert all(abs(norm[key] - 1) < 0.002 for key in columns if key.startswith("ps_"))
        assert norm["pdfUp"] - 1 == pytest.approx(1 - norm["pdfDown"], abs=1e-4)
        assert len(signal_theory_variations(era, SIGNAL_THEORY)) == 10
    monkeypatch.delitem(systematics.SIGNAL_THEORY_NORM, "2017")
    with pytest.raises(AnalysisError, match="no inclusive normalisation for 2017"):
        signal_theory_variations("2017", ["qcd_scale"])


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_the_stage_columns_exist(channel):
    """S0+ reads n-tuple columns of the v7b production (the top-pT weight, the b-pair flavour of the re-skim); S1 reads
    the columns of the shift production P6 only, on the signal."""
    on = analysis(crown_shifts=True, nn_friend=True, fit_categories="nn", **STAGES)
    definition = on.channel(channel)
    friends = NN_COLUMNS | {"fake_factor", "fake_factor_1", "fake_factor_2"}
    for sample in definition.samples:
        available = embedding_branches(channel) if sample.kind == "embedding" else branches(channel) | {"npartons"}
        missing = needed_columns(definition, sample) - available - friends
        assert missing == (P6_COLUMNS if sample.group == SIGNAL else set()), (sample.nick, missing)
