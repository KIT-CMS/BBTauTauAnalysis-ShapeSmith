"""The runs of the productions before v8 are what they were (v8 spec SH25, 9.3): the shift model of v8 sits behind
switches whose defaults are the behaviour of before.

- Every run YAML builds the Analysis it built before the shift model: tests/test_run_snapshot.py, whose digests are
  pinned on analysis 8c60aae and core 2146f73.
- A nominal-only run on a production with shifts (shift_set v8 with crown_shifts false: the S0 cards of v8) declares
  every shift for the skim and is otherwise the run it was: the same content, and the same datacards from the same
  histograms.
"""
import json

import pytest
from shapesmith.datacards import write_datacard
from shapesmith.estimates import run_estimates
from shapesmith.fill import bookings
from shapesmith.histogram import NOMINAL_VARIATION, HistKey, Histogram, HistogramSet
from shapesmith.model import ColumnVariation, declared_only

from bbtautau_shapesmith.constants import ERAS
from tests import snapshot
from tests.test_run2 import run2, tau  # noqa: F401 (the fixture)

PINNED = json.loads(snapshot.FIXTURE.read_text())
# the tau runs of a nominal-only production: crown_shifts false in the YAML
NOMINAL_ONLY = [(name, era) for name, era in snapshot.runs() if snapshot.switches_of(name).get("crown_shifts") is False]
V7B_FF = {"embedding": ["et", "mt"], "study_uncertainties": True, "dy_correction": "njets_btag", "fit_categories": "control", "symmetrise_ff_shifts": True}


def test_the_pinned_runs_are_those_before_the_shift_model():
    assert "8c60aae" in PINNED["state"] and "2146f73" in PINNED["state"]
    assert len(NOMINAL_ONLY) == 64 and {name for name, _ in NOMINAL_ONLY} >= {"smrun2_binned_v7b_ff", "smrun2_binned_v7b_ff_w", "smrun2_binned_v7b_ff_nn_m0_v1", "smrun2_binned_v7b"}


@pytest.mark.parametrize("name,era", NOMINAL_ONLY, ids=[f"{name}-{era}" for name, era in NOMINAL_ONLY])
def test_a_nominal_only_run_on_the_v8_production_only_declares_the_shifts(name, era):
    """-s switches.shift_set=v8 on a run with crown_shifts false: the run it was, plus the declared-only shifts."""
    on_v8 = snapshot.run_content(name, era, "switches.shift_set=v8")
    declared = [v for channel in on_v8["channels"].values() for v in channel["variations"] if v["type"] == "ColumnVariation" and v["regions"] == []]
    assert snapshot.digest(snapshot.without_declared_only(on_v8)) == PINNED["runs"][f"{name}/{era}"]
    assert bool(declared) == snapshot.switches_of(name).get("shape_systematics", True) and all(v["suffix"].startswith("__") and "required" not in v for v in declared)


def _histograms(analysis) -> HistogramSet:
    """One two-bin histogram for everything `hist` books for the categories: the nominal and every variation a
    booking is filled with (declared-only ones never), the data large enough for a positive jet-fake estimate."""
    hset = HistogramSet()
    for name, channel in analysis.channels.items():
        for booking in bookings(channel, True):
            scale = 100.0 if booking.process.role == "data" else 1.0
            for region in booking.regions:
                variations = [v.name for v in booking.variations if not declared_only(v) and (v.regions is None or region in v.regions)]
                for category in channel.categories:
                    for index, variation in enumerate([NOMINAL_VARIATION, *variations]):
                        first = 10.0 if variation == NOMINAL_VARIATION else 10.0 + (0.3 + 0.01 * (index % 7)) * (1 if variation.endswith("Up") else -1)
                        hset[HistKey(name, category.name, booking.process.name, region, variation, category.variable.name)] = Histogram([0.5, 1.5], [scale * first], [0.1])
    return hset


@pytest.fixture
def shapes(monkeypatch):
    """What datacards hand to the shapes file, per file: (bin, histogram name) -> (values, variances). Writing and
    reading back the several thousand histograms of a card takes seconds and tests nothing of the analysis."""
    from shapesmith import datacards

    written = {}

    def capture(path, entries):
        written[str(path)] = {(directory, name): (h.values.tolist(), h.variances.tolist()) for directory, name, h in entries}

    monkeypatch.setattr(datacards, "write_shapes", capture)
    return written


@pytest.mark.parametrize("era", ERAS)
@pytest.mark.parametrize("wjets", [False, True])
def test_the_s0_cards_of_a_v8_run_are_the_v7b_cards(run2, tmp_path, shapes, era, wjets):
    """The v7b fake-factor run and the same run on the v8 production with its shifts switched off: the same variations
    but for the declared-only shifts, the same estimators and lnN, and from the same histograms the same card text,
    template changes and shapes."""
    switches = {**V7B_FF, "ff_processes": ["QCD", "Wjets", "ttbar"] if wjets else ["QCD", "ttbar"]}
    v7b = tau(run2, era, **switches)
    s0 = tau(run2, era, **switches, shift_set="v8")
    assert snapshot.content(s0) != snapshot.content(v7b)
    assert snapshot.without_declared_only(snapshot.content(s0)) == snapshot.content(v7b)
    for name, channel in s0.channels.items():
        added = [v for v in channel.variations if v not in v7b.channel(name).variations]
        assert added and all(isinstance(v, ColumnVariation) and declared_only(v) and v.suffix == f"__{v.name}" for v in added)
        assert [v for v in channel.variations if v not in added] == list(v7b.channel(name).variations)
        assert channel.estimators == v7b.channel(name).estimators and channel.processes == v7b.channel(name).processes
    assert s0.lnn == v7b.lnn and s0.symmetrised_variations == v7b.symmetrised_variations and s0.lnn_variations == v7b.lnn_variations
    cards = {}
    for label, analysis in (("v7b", v7b), ("s0", s0)):
        hset = _histograms(analysis)
        run_estimates(hset, analysis, list(analysis.channels))
        cards[label] = write_datacard(hset, analysis, list(analysis.channels), tmp_path / label / "all", min_background=None)
    assert cards["s0"].read_text() == cards["v7b"].read_text() and " shape " in cards["v7b"].read_text() and "CMS_ff_" in cards["v7b"].read_text()
    assert (cards["s0"].parent / "template_changes.json").read_text() == (cards["v7b"].parent / "template_changes.json").read_text()
    old, new = (shapes[str(cards[label].parent / "common" / f"htt_input_{era}.root")] for label in ("v7b", "s0"))
    assert old == new and len(old) > 1000 and any("CMS_ff_" in name for _, name in old) and any(name == "data_obs" for _, name in old)
