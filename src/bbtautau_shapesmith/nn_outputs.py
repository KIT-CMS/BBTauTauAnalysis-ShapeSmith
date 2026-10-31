"""The validation of the NN outputs (nn_limit_spec.md 7.1, NV4 and NV4b; switch nn_outputs): the outputs of the model
as control variables, data against the prediction (shapesmith input-gof), the MC jet fakes in AR x FF against the SR
(scripts/nn_outputs_transfer.py) and the fake-factor fractions per NN category (scripts/nn_outputs_fractions.py).

Control variables (the scores in the 1000 fine bins of the NN categories, merged by input-gof and the scripts):
- NN_cat_<category>: the score (<prefix>predicted_max_value) in each NN category of the scheme (k1: the class and
  n_bjets < 3; k2: the HH category per b-tag bin; thesis: the class), every other event below the axis. An HH category
  hides its events with a score at or above the blinding edge of the run's cards (switch nn_blinding: their
  blinding.json, the HH bins `htt_<ch>_<k>hi_<era>`) in the nominal region and in the regions of its estimates, the
  events NN_score leaves out (variables.nn_score_blinding); there it holds the `lo` part only. Without nn_blinding an HH
  category hides every event there. Other regions (same sign, lepton anti-isolation, high mT, ...) keep every event.
- NN_P_<class>: the score of each class (<prefix>nn_score_<class>) over every event, without the events NN_score
  leaves out (every event predicted as the signal class in the nominal region and the regions of its estimates).
- NN_<input>_<category> for the inputs INPUTS (n_jets: its repair by the S0+ ttbar shape; mass_2: its FF transfer,
  the NV1 decisions of 2026-10-27): the input in each category, the HH categories hidden as NN_cat_.
- with ff_transfer (the MC jet fakes in AR x FF against the SR, NV4 FF closure): MC-only copies NNmc_cat_<HH
  category>, NNmc_P_<class> and NNmc_<input>_<HH category> of the variables that hide events; data and embedding
  below the axis (is_mc), every simulated event of an HH category or class.
- NN_ar_<HH category> (NV4b): the score of an HH category in events whose tau (lt) or one of whose taus (tt) fails the
  nominal working point, so never an event of the signal region; for the fractions in the anti-isolated region.

NV4b regions (switch nn_ff_payload, the fake-factor payload of the friend: {year} and {channel} filled):
anti_iso_ttbar_fraction and anti_iso_ttbar_fraction_up, the anti-isolated region (no fake factor) weighted by the
ttbar fraction of the payload (process_fractions, nominal and frac_ttbar_J_up; in tt the fractions of the
anti-isolated leg), so that the fraction the fake factor applies per category and score bin is their ratio to the
unweighted anti-isolated region (abcd_anti_iso in mc mode).

Blinding (spec 6.4): no variable shows data of an HH category above its blinding edge in the nominal region; NN_cat_
and NN_<input>_ of an HH category need the edges (nn_blinding) to show its lo part, NN_ar_ selects anti-isolated taus
only, the NNmc_ variables hold no data.
"""
from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from pathlib import Path

from shapesmith.model import AnalysisError, Region, Variable

from bbtautau_shapesmith.constants import LT_CHANNELS, NN_CLASS_COLUMN, NN_SCORE_COLUMN, TAU_LEGS, TAU_VS_JET_WP
from bbtautau_shapesmith.cuts import anti_iso_cut
from bbtautau_shapesmith.variables import _BINNING, NN_FINE_BINNING, NN_SIGNAL_CLASS, blinded_score, categories

INPUTS = ("n_jets", "mass_2")
SIMULATED = "(~is_mc)"  # the events below the axis of an MC-only variable: data and embedding
FRACTION_REGION = "anti_iso_ttbar_fraction"
FRACTION_PROCESS = "ttbar"  # the payload label of the ttbar fraction (frac_ttbar_J)
FRACTION_SYSTS = {FRACTION_REGION: "nominal", f"{FRACTION_REGION}_up": "frac_ttbar_J_up"}
# the fraction correction of each leg (tt: the leading and the subleading tau)
FRACTION_CORRECTIONS = {"lt": "process_fractions", 1: "process_fractions", 2: "process_fractions_subleading"}


def _edges(values) -> tuple[float, ...]:
    return tuple(float(round(v, 6)) for v in values)


def _below_axis(expr: str, hidden: str) -> str:
    """`expr` with the events of `hidden` moved below every axis (expr - 1000 for them)."""
    return f"{expr} - 1000 * ({hidden})"


def blinding_edges(path: str | Path, channel: str, era: str) -> dict[str, float]:
    """The blinding edge of every HH category of a channel from a blinding.json of the datacards (shapesmith masking):
    {category suffix of the card bin, e.g. '1' (k1) or '1' and '2' (k2): edge}."""
    content = json.loads(Path(path).read_text())
    prefix, suffix = f"htt_{channel}_", f"hi_{era}"
    edges = {}
    for name, entry in content.get("categories", {}).items():
        if name.startswith(prefix) and name.endswith(suffix):
            edges[name[len(prefix):-len(suffix)]] = float(entry["edge"])
    if not edges:
        raise AnalysisError(f"nn_blinding: {path} has no blinded HH category of {channel} {era}")
    return edges


def _hh_categories(classes: tuple[str, ...], scheme: str, prefix: str):
    """(index of the card bin, category) of every NN category; the card bins number the categories from 1."""
    return list(enumerate(categories(classes, scheme, "data_blind", prefix), start=1))


def output_variables(classes: tuple[str, ...], scheme: str, prefix: str, blinded: str, edges: Mapping[str, float] | None,
                     channel: str, ff_transfer: bool) -> dict[str, Variable]:
    """The control variables of the switch nn_outputs (module docstring); `blinded`: the events NN_score leaves out
    (variables.nn_score_blinding), `edges`: the blinding edge by card bin index (blinding_edges) or None."""
    score = f"{prefix}{NN_SCORE_COLUMN}"
    fine = _edges(NN_FINE_BINNING)
    variables = {}
    hh_hidden = {}
    for index, category in _hh_categories(classes, scheme, prefix):
        outside = f"~({category.cut})"
        if not category.name.startswith(NN_SIGNAL_CLASS):
            hidden = outside
        else:
            edge = None if edges is None else edges.get(str(index))
            if edges is not None and edge is None:
                raise AnalysisError(f"nn_blinding: no blinding edge for the HH category {category.name} (card bin {index}) of {channel}, got {dict(edges)}")
            hidden = f"{outside} | (({blinded}) & ({score} >= {edge!r}))" if edge is not None else f"{outside} | ({blinded})"
            hh_hidden[category.name] = outside
            # NV4b: anti-isolated taus only, never an event of the signal region
            variables[f"NN_ar_{category.name}"] = Variable(f"NN_ar_{category.name}", _below_axis(score, f"{outside} | ~({anti_iso_cut(channel)})"), fine)
        variables[f"NN_cat_{category.name}"] = Variable(f"NN_cat_{category.name}", _below_axis(score, hidden), fine)
        for name in INPUTS:
            variables[f"NN_{name}_{category.name}"] = Variable(f"NN_{name}_{category.name}", _below_axis(name, hidden), _edges(_BINNING[name]))
    for name in classes:
        column = f"{prefix}nn_score_{name}"
        variables[f"NN_P_{name}"] = Variable(f"NN_P_{name}", blinded_score(column, blinded), fine)
        if ff_transfer:
            variables[f"NNmc_P_{name}"] = Variable(f"NNmc_P_{name}", _below_axis(column, SIMULATED), fine)
    if ff_transfer:
        for category, outside in hh_hidden.items():
            hidden = f"{outside} | {SIMULATED}"
            variables[f"NNmc_cat_{category}"] = Variable(f"NNmc_cat_{category}", _below_axis(score, hidden), fine)
            for name in INPUTS:
                variables[f"NNmc_{name}_{category}"] = Variable(f"NNmc_{name}_{category}", _below_axis(name, hidden), _edges(_BINNING[name]))
    return variables


# -- NV4b: the fractions of the fake-factor payload ------------------------------------------------------------------


def _read(path: str | Path) -> dict:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def _number(value: float) -> str:
    return repr(float(value))


def node_expression(node, assignments: Mapping[str, str]) -> str:
    """A correctionlib (schema v2) node as an expression of the columns of its binning inputs: category nodes take the
    content of their input's value in `assignments` (else the default), binning nodes become a sum of indicators
    (flow clamp: the first and the last bin open; error: outside nothing), numbers stay."""
    if isinstance(node, (int, float)):
        return _number(node)
    kind = node.get("nodetype")
    if kind == "category":
        key = assignments.get(node["input"])
        for item in node["content"]:
            if key is not None and str(item["key"]) == str(key):
                return node_expression(item["value"], assignments)
        if node.get("default") is None:
            raise AnalysisError(f"payload: no content for {node['input']} = {key!r} and no default")
        return node_expression(node["default"], assignments)
    if kind == "binning":
        column, edges, flow = node["input"], [float(e) for e in node["edges"]], node.get("flow", "error")
        if flow not in ("clamp", "error"):
            raise AnalysisError(f"payload: binning flow {flow!r} of {column} is not supported (clamp or error)")
        terms = []
        for i, content in enumerate(node["content"]):
            low = None if flow == "clamp" and i == 0 else f"({column} >= {_number(edges[i])})"
            high = None if flow == "clamp" and i == len(edges) - 2 else f"({column} < {_number(edges[i + 1])})"
            condition = " & ".join(part for part in (low, high) if part) or "True"
            terms.append(f"({condition}) * {node_expression(content, assignments)}")
        return f"({' + '.join(terms)})"
    raise AnalysisError(f"payload: node type {kind!r} is not supported")


def fraction_expression(payload: str | Path, channel: str, process: str = FRACTION_PROCESS, syst: str = "nominal") -> str:
    """The fraction of `process` the fake-factor payload of a channel (fake_factors_<channel>.json.gz) applies to an
    event of the anti-isolated region, as an expression: lt the fractions of the tau; tt those of the leg that fails
    the nominal working point (the other passes it in the anti-isolated region). `syst`: nominal or a variation of
    the payload without its correction-name prefix, e.g. frac_ttbar_J_up."""
    corrections = {c["name"]: c for c in _read(payload)["corrections"]}

    def leg(name: str) -> str:
        if name not in corrections:
            raise AnalysisError(f"nn_ff_payload: {payload} has no correction {name}")
        correction = corrections[name]
        return node_expression(correction["data"], {"process": process, "syst": f"{name}{syst}"})

    if channel in LT_CHANNELS:
        return leg(FRACTION_CORRECTIONS["lt"])
    return " + ".join(f"(id_tau_vsJet_{TAU_VS_JET_WP}_{n} < 0.5) * {leg(FRACTION_CORRECTIONS[n])}" for n in TAU_LEGS[channel])


def fraction_regions(payload: str | Path, channel: str) -> tuple[Region, ...]:
    """The NV4b regions (module docstring): the anti-isolated region weighted by the payload's ttbar fraction."""
    return tuple(Region(name, replace_cuts={"tau_iso": anti_iso_cut(channel)}, add_weights={"ttbar_fraction": fraction_expression(payload, channel, syst=syst)})
                 for name, syst in FRACTION_SYSTS.items())
