"""The input validation of the NN phase (nn_limit_spec.md 7.1, NV1-NV3; nn_validation.py): the union of the NN inputs and
the models that read them, their control variables in fine bins with the over- and underflow in the edge bins, the
collinear mass, the eta-phi occupancy cells, the b-tag working points, the preliminary pseudo-shifts against an
independent four-vector computation, the switches and the run YAMLs of the validation."""
import gzip
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from shapesmith.config import FriendConfig, load_analysis, load_config
from shapesmith.expressions import evaluate
from shapesmith.input_gof import is_integer_binning
from shapesmith.model import AnalysisError, ColumnVariation
from shapesmith.skim import needed_columns

from bbtautau_shapesmith import nn_validation as v
from bbtautau_shapesmith.analysis import build
from bbtautau_shapesmith.constants import ERAS, TAU_CHANNELS
from bbtautau_shapesmith.dilepton import build as build_dilepton
from bbtautau_shapesmith.ml import THESIS_INPUTS
from tests.helpers import DATABASE, EMBEDDING_LIST, REPO, branches, config, embedding_branches

# the step-2 schema of bbtautau-nn (schema.py SLOTS and STEP2_HL, 327645e): the export columns of the token fields and the
# HL features besides the thesis inputs
STEP2_TOKENS = (
    "pt_1", "eta_1", "phi_1", "mass_1", "iso_1", "tau_decaymode_1", "pt_2", "eta_2", "phi_2", "mass_2", "tau_decaymode_2",
    "bpair_pt_1", "bpair_eta_1", "bpair_phi_1", "bpair_mass_1", "bpair_btag_value_1", "bpair_pt_2", "bpair_eta_2", "bpair_phi_2", "bpair_mass_2",
    "bpair_btag_value_2", "jpt_1", "jeta_1", "jphi_1", "jtag_value_1", "jpt_2", "jeta_2", "jphi_2", "jtag_value_2", "met", "metphi", "metSumEt",
    "pt_fastmtt", "eta_fastmtt", "phi_fastmtt", "m_fastmtt",
)
STEP2_EXTRA_HL = ("mt_1", "mt_2", "pzetamissvis", "mass_tautaubb", "pt_tautaubb")
VALIDATION = "/ceph/sdaigler/shapesmith_test/nn-validate"
FF_SWITCHES = {"nn_validation": True, "pseudo_shifts": True}


def test_the_union_of_the_inputs_and_the_models_that_read_them():
    read_by = lambda model: {name for name, uses in v.NN_INPUTS.items() if model in uses and name not in v.COMPUTED}  # noqa: E731
    # the thesis recipe and M0: the 19 columns of the thesis inputs and the legs and MET of mtt_coll_approx, 28 columns
    assert read_by("M0") == read_by("R3-R5") == read_by("M1") == set(THESIS_INPUTS) | set(v.COLLINEAR_INPUTS) and len(read_by("M0")) == 28
    assert read_by("M2/M3") == set(STEP2_TOKENS) | set(THESIS_INPUTS) | set(STEP2_EXTRA_HL) | {"deltaR_ditaupair", "bpair_deltaR", "mt_tot"}
    assert set(v.NN_INPUTS) == read_by("M0") | read_by("M2/M3") | {"mtt_coll_approx"} and v.NN_INPUTS["mtt_coll_approx"] == ("R3-R5", "M0", "M1", "M2/M3")
    assert not any("hadronFlavour" in name or "vsJet" in name for name in v.NN_INPUTS)


def test_no_raw_vsjet_score_and_no_constant_is_tested():
    assert "iso_1" not in v.inputs("tt") and "iso_1" in v.inputs("mt") and "iso_1" in v.inputs("et")
    assert "mass_1" not in v.inputs("mt") and "mass_1" in v.inputs("et") and "mass_1" in v.inputs("tt")
    assert "tau_decaymode_1" in v.inputs("tt") and not {"tau_decaymode_1"} & (set(v.inputs("et")) | set(v.inputs("mt")))
    assert set(v.inputs("tt")) == set(v.NN_INPUTS) - {"iso_1"}


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_every_input_is_a_variable_with_its_under_and_overflow_in_the_edge_bins(channel):
    variables = v.validation_variables(channel)
    assert set(v.inputs(channel)) | {"yield"} | {f"occupancy_{name}" for name in v.OCCUPANCY} == set(variables)
    assert set(v.validation_variables(channel, fastmtt=False)) == set(variables) - {"m_fastmtt", "pt_fastmtt", "eta_fastmtt", "phi_fastmtt"}
    for name in v.inputs(channel):
        variable = variables[name]
        edges = np.asarray(variable.edges)
        assert is_integer_binning(edges) == (name in ("n_jets", "n_bjets", "tau_decaymode_1", "tau_decaymode_2")), name
        if name == "mtt_coll_approx":
            continue
        values = np.array([-1e6, edges[0], 0.5 * (edges[0] + edges[-1]), edges[-1], 1e7, np.nan])
        clipped = evaluate(pd.DataFrame({name: values.astype(np.float32)}), variable.expr)
        counts, _ = np.histogram(clipped[:-1], bins=edges)
        assert counts[0] == 2 and counts[-1] == 2 and counts.sum() == 5 and np.isnan(clipped[-1]), name
        assert clipped[2] == pytest.approx(values[2], rel=1e-6), name


def collinear_mass(c: dict) -> np.ndarray:
    """The old CROWN collinear mass (bbtautau-nn schema.mtt_coll_approx, in float64): a negative fraction flips the
    momentum and keeps the energy positive; -10 for |sin(phi_2 - phi_1)| < 1e-6."""
    sin = np.sin(c["phi_2"] - c["phi_1"])
    with np.errstate(all="ignore"):
        a1 = 1 + c["met"] / c["pt_1"] * np.sin(c["phi_2"] - c["metphi"]) / sin
        a2 = 1 + c["met"] / c["pt_2"] * np.sin(c["phi_1"] - c["metphi"]) / -sin
        vectors = [a[:, None] * four_vector(c[f"pt_{i}"], c[f"eta_{i}"], c[f"phi_{i}"], c[f"mass_{i}"]) for a, i in ((a1, 1), (a2, 2))]
        for vector in vectors:
            vector[:, 3] = np.abs(vector[:, 3])
        return np.where(np.abs(sin) >= 1e-6, mass(vectors[0] + vectors[1]), -10.0)  # the guarded events give inf - inf


def four_vector(pt, eta, phi, m) -> np.ndarray:
    return np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m**2)], axis=1)


def mass(p) -> np.ndarray:
    return np.sqrt(np.maximum(p[:, 3] ** 2 - p[:, 0] ** 2 - p[:, 1] ** 2 - p[:, 2] ** 2, 0.0))


def transverse(p) -> np.ndarray:
    return np.hypot(p[:, 0], p[:, 1])


def events(n: int = 3000, seed: int = 1) -> dict[str, np.ndarray]:
    """Random legs, b pair, leading jets (in 1/3 the first b jet, in 1/3 the second, else another jet; the second leading
    jet the second b jet in 1/3), MET and FastMTT."""
    rng = np.random.default_rng(seed)
    c = {}
    for prefix, low, high, mass_range in (("", 25, 200, (0.1, 1.8)), ("bpair_", 20, 300, (2, 30)), ("j", 20, 300, (2, 30))):
        for i in (1, 2):
            pt, eta = ("jpt", "jeta") if prefix == "j" else (f"{prefix}pt", f"{prefix}eta")
            c[f"{pt}_{i}"], c[f"{eta}_{i}"] = rng.uniform(low, high, n), rng.uniform(-2.3, 2.3, n)
            c[f"{'jphi' if prefix == 'j' else prefix + 'phi'}_{i}"] = rng.uniform(-np.pi, np.pi, n)
            if prefix != "j":
                c[f"{prefix}mass_{i}"] = rng.uniform(*mass_range, n)
    c["choice"] = rng.integers(0, 3, n)
    for k in ("pt", "eta", "phi"):
        c[f"j{k}_1"] = np.select([c["choice"] == 0, c["choice"] == 1], [c[f"bpair_{k}_1"], c[f"bpair_{k}_2"]], c[f"j{k}_1"])
        c[f"j{k}_2"] = np.where(c["choice"] == 0, c[f"bpair_{k}_2"], c[f"j{k}_2"])
    c["met"], c["metphi"] = rng.uniform(1, 150, n), rng.uniform(-np.pi, np.pi, n)
    c["pt_fastmtt"], c["phi_fastmtt"], c["m_fastmtt"] = rng.uniform(5, 200, n), rng.uniform(-np.pi, np.pi, n), rng.uniform(50, 300, n)
    return c


def reference(c: dict, f1: float = 1.0, f2: float = 1.0, jets: float = 1.0) -> dict[str, np.ndarray]:
    """The quantities of the scaled four momenta with the MET propagated, computed from four-vectors."""
    legs = [four_vector(f * c[f"pt_{i}"], c[f"eta_{i}"], c[f"phi_{i}"], f * c[f"mass_{i}"]) for f, i in ((f1, 1), (f2, 2))]
    b = [four_vector(jets * c[f"bpair_pt_{i}"], c[f"bpair_eta_{i}"], c[f"bpair_phi_{i}"], jets * c[f"bpair_mass_{i}"]) for i in (1, 2)]
    distinct = [c["choice"] >= 2, c["choice"] >= 1]  # a leading jet that is no b-pair jet
    change = sum((f - 1) * four_vector(c[f"pt_{i}"], c[f"eta_{i}"], c[f"phi_{i}"], c[f"mass_{i}"]) for f, i in ((f1, 1), (f2, 2)))
    jet_change = (jets - 1) * (sum(four_vector(c[f"bpair_pt_{i}"], c[f"bpair_eta_{i}"], c[f"bpair_phi_{i}"], 0 * c["met"]) for i in (1, 2))
                               + sum(d[:, None] * four_vector(c[f"jpt_{i}"], c[f"jeta_{i}"], c[f"jphi_{i}"], 0 * c["met"]) for i, d in zip((1, 2), distinct)))
    mx = c["met"] * np.cos(c["metphi"]) - change[:, 0] - jet_change[:, 0]
    my = c["met"] * np.sin(c["metphi"]) - change[:, 1] - jet_change[:, 1]
    met, metphi = np.hypot(mx, my), np.arctan2(my, mx)
    met_vector = np.stack([mx, my, 0 * mx, met], axis=1)

    def mt(pt, phi):
        return np.sqrt(2 * pt * met * (1 - np.cos(phi - metphi)))

    result = {"met": met, "metphi": metphi, "m_vis": mass(legs[0] + legs[1]), "pt_vis": transverse(legs[0] + legs[1]),
              "mt_1": mt(f1 * c["pt_1"], c["phi_1"]), "mt_2": mt(f2 * c["pt_2"], c["phi_2"]), "bpair_m_inv": mass(b[0] + b[1]), "bpair_pt_dijet": transverse(b[0] + b[1])}
    mixed = 2 * f1 * c["pt_1"] * f2 * c["pt_2"] * (1 - np.cos(c["phi_1"] - c["phi_2"]))
    result["mt_tot"] = np.sqrt(result["mt_1"] ** 2 + result["mt_2"] ** 2 + mixed)
    zeta = np.stack([np.cos(c["phi_1"]) + np.cos(c["phi_2"]), np.sin(c["phi_1"]) + np.sin(c["phi_2"])], axis=1)
    zeta /= np.linalg.norm(zeta, axis=1)[:, None]
    visible = (legs[0] + legs[1])[:, :2]
    result["pzetamissvis"] = (np.stack([mx, my], axis=1) * zeta).sum(1) - v.PZETA_ALPHA * (visible * zeta).sum(1)
    total = legs[0] + legs[1] + b[0] + b[1] + met_vector
    result["mass_tautaubb"], result["pt_tautaubb"] = mass(total), transverse(total)
    fastmtt = np.stack([c["pt_fastmtt"] * np.cos(c["phi_fastmtt"]), c["pt_fastmtt"] * np.sin(c["phi_fastmtt"])], axis=1) - v.FASTMTT_MET_FRACTION * jet_change[:, :2]
    result["pt_fastmtt"] = transverse(fastmtt) * (1 + v.FASTMTT_TES_PT * (f1 - 1 + f2 - 1))
    return result


@pytest.mark.parametrize("label,legs,jets", [("jes", {}, 1.03), ("jes down", {}, 0.97), ("tes lt", {2: 0.985}, 1.0), ("tes tt", {1: 1.015, 2: 1.015}, 1.0)])
def test_the_pseudo_shifts_against_four_vectors(label, legs, jets):
    """Every derived column of a pseudo-shift equals the quantity of the scaled four momenta with the MET propagated."""
    c = events()
    nominal = reference(c)
    frame = pd.DataFrame({**c, **{name: nominal[name] for name in ("m_vis", "pt_vis", "mt_1", "mt_2", "mt_tot", "pzetamissvis", "mass_tautaubb", "pt_tautaubb", "bpair_m_inv", "bpair_pt_dijet")},
                          "mjj": 100.0 + 0 * c["met"], "pt_dijet": 50.0 + 0 * c["met"]})
    derived = v.pseudo_shift(legs, jets)
    shifted = reference(c, legs.get(1, 1.0), legs.get(2, 1.0), jets)
    compared = set(shifted) & set(derived)
    assert compared >= {"met", "metphi", "mt_1", "mt_2", "mt_tot", "pzetamissvis", "mass_tautaubb", "pt_tautaubb"}
    for column in compared:
        assert np.allclose(evaluate(frame, derived[column]), shifted[column], rtol=1e-6, atol=1e-6), column
    for column in ("bpair_pt_1", "jpt_2", "bpair_mass_2", "mjj"):
        assert (column in derived) == (jets != 1.0)
    for leg in (1, 2):
        assert (f"pt_{leg}" in derived) == (leg in legs) and (f"mass_{leg}" in derived) == (leg in legs)
    # FastMTT: the mean response of a refit (module docstring)
    m_ratio = evaluate(frame, derived["m_fastmtt"]) / c["m_fastmtt"]
    expected = (shifted["m_vis"] / nominal["m_vis"]) ** v.FASTMTT_TES_MASS_POWER if legs else 1.0 + v.FASTMTT_JES_MASS * (jets - 1.0)
    assert np.allclose(m_ratio, expected, rtol=1e-6) and "eta_fastmtt" not in derived


def test_the_collinear_mass_is_the_old_crown_formula():
    c = events(seed=4)
    c["phi_2"][:5] = c["phi_1"][:5]  # back-to-back or collinear legs: the default
    frame = pd.DataFrame(c)
    got = evaluate(frame, v.COLLINEAR_MASS)
    assert np.allclose(got, collinear_mass(c), rtol=1e-6) and np.array_equal(got[:5], [-10.0] * 5)
    shifted = evaluate(frame, v.COLLINEAR_MASS.replace("met", "(1.5 * met)").replace("(1.5 * met)phi", "metphi"))
    assert not np.allclose(shifted, got)


def test_the_occupancy_cells():
    expr = v.occupancy_expression("eta", "phi")
    frame = pd.DataFrame({"eta": [-2.49, -2.5, 0.01, 2.49, 3.5, -1.4, -10.0], "phi": [-math.pi, 0.0, 0.01, 3.14, -0.1, -1.2, -10.0]})
    cells = evaluate(frame, expr)
    n_phi = v.OCCUPANCY_N_PHI
    # |eta| > 2.5 in the edge cells; HEM; an absent object (CROWN default -10) outside the histogram
    assert cells.tolist() == [0.0, 12.0, 10 * n_phi + 12, 19 * n_phi + 23, 19 * n_phi + 11, 4 * n_phi + 7, -1000.0]
    edges = v.validation_variables("mt")["occupancy_jet_1"].edges
    assert edges[0] == -0.5 and edges[-1] == v.OCCUPANCY_N_ETA * n_phi - 0.5


def test_the_btag_working_points_are_those_of_the_payloads_crown_pins():
    for era, (medium, tight) in v.BTAG_WP.items():
        path = Path(f"/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run2-{era}-UL-NanoAODv15/2026-06-18/btagging.json.gz")
        if not path.exists():
            pytest.skip("no cvmfs")
        correction = next(c for c in json.loads(gzip.open(path).read())["corrections"] if c["name"] == "UParTAK4_wp_values")
        points = {item["key"]: item["value"] for item in correction["data"]["content"]}
        assert (points["M"], points["T"]) == (medium, tight), era
    assert set(v.BTAG_WP) == set(ERAS)


def analysis(**switches):
    switches = {"jet_fakes": "ff", "embedding": ["et", "mt"], "sample_lists": ["sm2018_binned_v3", EMBEDDING_LIST], **switches}
    return build(config(TAU_CHANNELS, **switches))


@pytest.mark.parametrize("pseudo_shifts", [False, True])
def test_the_validation_switches_of_the_tau_run(pseudo_shifts):
    on = analysis(fastmtt_friend=True, nn_validation=True, pseudo_shifts=pseudo_shifts)
    off = analysis(fastmtt_friend=True)
    for name in TAU_CHANNELS:
        channel = on.channel(name)
        assert set(channel.variables) == set(v.validation_variables(name)) and set(v.FASTMTT_INPUTS) <= set(channel.keep_columns)
        pseudo = [variation for variation in channel.variations if "pseudo" in variation.name]
        expected = {f"{prefix}_2018{d}" for prefix in (v.PSEUDO_JES_NAME, v.PSEUDO_TES_NAME, *((v.PSEUDO_EMB_TES_NAME,) if name != "tt" else ())) for d in ("Up", "Down")}
        assert {variation.name for variation in pseudo} == (expected if pseudo_shifts else set())
        assert all(isinstance(variation, ColumnVariation) and variation.applies_to == (("embedding",) if "emb" in variation.name else ("mc",)) for variation in pseudo)
        assert [r.name for r in channel.regions] == [r.name for r in off.channel(name).regions]
        assert [x.name for x in channel.variations if "pseudo" not in x.name] == [x.name for x in off.channel(name).variations]
    assert set(on.symmetrised_variations) - set(off.symmetrised_variations) == ({"CMS_pseudo_jes_2018", "CMS_pseudo_tes_2018", "CMS_pseudo_emb_tes_2018"} if pseudo_shifts else set())


def test_ff_transfer_adds_the_fake_factor_region_and_its_shifts_in_mc_mode():
    on = analysis(jet_fakes="mc", fastmtt_friend=True, control_regions=True, nn_validation=True, ff_transfer=True)
    off = analysis(jet_fakes="mc", fastmtt_friend=True, control_regions=True, nn_validation=True)
    ff = analysis(fastmtt_friend=True, crown_shifts=False, study_uncertainties=True)
    for name in TAU_CHANNELS:
        region = on.channel(name).region("anti_iso")
        assert region == ff.channel(name).region("anti_iso")  # the jetFakes region of the ff mode
        assert "anti_iso" not in [r.name for r in off.channel(name).regions]
        shifts = {x.name for x in on.channel(name).variations} - {x.name for x in off.channel(name).variations}
        assert shifts == {x.name for x in ff.channel(name).variations if x.name.startswith("CMS_ff_")} and shifts
    with pytest.raises(AnalysisError, match="ff_transfer requires jet_fakes: mc"):
        analysis(ff_transfer=True)
    with pytest.raises(AnalysisError, match="pseudo_shifts requires nn_validation"):
        analysis(pseudo_shifts=True)


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_the_validation_columns_exist(channel):
    """Every column the validation run reads is an n-tuple branch (v7 sm_config) or a friend column."""
    run = analysis(fastmtt_friend=True, nn_validation=True, pseudo_shifts=True)
    definition = run.channel(channel)
    friends = {"fake_factor", "fake_factor_1", "fake_factor_2", "m_fastmtt", "pt_fastmtt", "eta_fastmtt", "phi_fastmtt"}
    for sample in definition.samples:
        available = embedding_branches(channel) if sample.kind == "embedding" else branches(channel) | {"npartons"}
        assert not needed_columns(definition, sample) - available - friends, sample.nick


def test_the_dilepton_validation_has_two_jets_and_the_b_tag_scores():
    on = build_dilepton(config(("em", "mm", "ee"), analysis="bbtautau_shapesmith.dilepton:build", sample_lists=["sm2018_binned_v3"], nn_validation=True))
    off = build_dilepton(config(("em", "mm", "ee"), analysis="bbtautau_shapesmith.dilepton:build", sample_lists=["sm2018_binned_v3"]))
    for name in ("em", "mm", "ee"):
        assert on.channel(name).cuts["jets"] == "n_jets >= 2" and off.channel(name).cuts["jets"] == ("n_jets >= 2" if name == "em" else "True")
        assert set(v.BTAG_SCORES) <= set(on.channel(name).variables) and set(v.FASTMTT_INPUTS) <= set(on.channel(name).keep_columns)
        for sample in on.channel(name).samples:
            assert needed_columns(on.channel(name), sample) <= branches(name) | {"npartons"}, sample.nick


# -- the run YAMLs ---------------------------------------------------------------------------------------------------

STORE = "/store/user/sdaigler/CROWN/ntuples"
VALIDATION_RUNS = {  # run YAML -> (the run it copies, the switches it adds, its directory under VALIDATION)
    "smrun2_binned_v7b_ff_nn_validation": ("smrun2_binned_v7b_ff_nn", FF_SWITCHES, "tau_ff_nn"),
    "smrun2_binned_v7b_nn_validation_control": ("smrun2_binned_v7b", {**FF_SWITCHES, "ff_transfer": True}, "tau_control_nn"),
    "smrun2_binned_v7b_dilepton_nn_validation": ("smrun2_binned_v7b_dilepton", {"nn_validation": True}, "dilepton_nn"),
}


def load(name: str, era: str):
    return load_config(REPO / "configs" / f"{name}.yaml", [f"era={era}", f"sample_database={DATABASE}"])


@pytest.mark.parametrize("name", VALIDATION_RUNS)
@pytest.mark.parametrize("era", ERAS)
def test_each_validation_run_is_its_run_with_the_validation_switches(name, era):
    copied, switches, directory = VALIDATION_RUNS[name]
    run, base = load(name, era), load(copied, era)
    assert (run.skim_dir, run.output_dir) == (Path(VALIDATION) / "skims" / directory / era, Path(VALIDATION) / "output" / directory / era)
    expected = {key: value for key, value in json.loads(base.model_dump_json()).items() if key not in ("skim_dir", "output_dir", "ml_dir", "web", "log_level", "combine")}
    actual = {key: value for key, value in json.loads(run.model_dump_json()).items() if key not in ("skim_dir", "output_dir", "ml_dir", "web", "log_level", "combine")}
    expected["switches"].update(switches)
    for key in ("ml_labels", "ml_variables"):
        expected["switches"].pop(key, None)
    if name == "smrun2_binned_v7b_ff_nn_validation":  # the tau channels only: em is a control channel of the fits
        expected["channels"] = list(TAU_CHANNELS)
    if name == "smrun2_binned_v7b_nn_validation_control":  # the fake factor of ff_v7b for the anti-isolated region
        expected["ntuples"]["friends"] = [
            FriendConfig(base=f"{STORE}/smrun2_binned_v7/CROWNFriend/ff_v7b", applies_to=("data",), channels=TAU_CHANNELS).model_dump(mode="json"),
            FriendConfig(base=f"{STORE}/smrun2_binned_v7b/CROWNFriend/ff_v7b", applies_to=("mc", "embedding"), channels=TAU_CHANNELS).model_dump(mode="json"),
            *expected["ntuples"]["friends"],
        ]
    assert actual == expected
    assert run.ml_dir is None and run.web is None and run.workers <= 16
    analysis = load_analysis(run)
    assert all(not any("NN_score" == variable for variable in channel.variables) for channel in analysis.channels.values())


def test_the_validation_runs_write_nowhere_else():
    for name in VALIDATION_RUNS:
        raw = yaml.safe_load((REPO / "configs" / f"{name}.yaml").read_text())
        assert raw["skim_dir"].startswith(VALIDATION) and raw["output_dir"].startswith(VALIDATION) and "ml_dir" not in raw


def test_the_fastmtt_build_takes_the_crown_sources_with_local_includes(tmp_path):
    """scripts/fastmtt_refit.py copies src/SVFit and include/SVFit of a CROWN checkout with local includes, the stub
    logger and the driver; a second preparation leaves the files untouched (ACLiC rebuilds on newer sources)."""
    spec = importlib.util.spec_from_file_location("fastmtt_refit", REPO / "scripts" / "fastmtt_refit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    crown = tmp_path / "crown"
    for name in module.SOURCES:
        for kind, suffix in (("src", "cxx"), ("include", "hxx")):
            (crown / kind / "SVFit").mkdir(parents=True, exist_ok=True)
            (crown / kind / "SVFit" / f"{name}.{suffix}").write_text(f'#include "../../include/SVFit/{name}.hxx"\n#include "../../include/utility/Logger.hxx"\n')
    driver = module.prepare(crown, tmp_path / "build")
    assert driver.read_text() == module.DRIVER and (tmp_path / "build" / "Logger.hxx").read_text() == module.LOGGER
    assert (tmp_path / "build" / "FastMTT.cxx").read_text() == '#include "FastMTT.hxx"\n#include "Logger.hxx"\n'
    stamp = (tmp_path / "build" / "FastMTT.cxx").stat().st_mtime_ns
    module.prepare(crown, tmp_path / "build")
    assert (tmp_path / "build" / "FastMTT.cxx").stat().st_mtime_ns == stamp
    assert module.LEG_TYPES["em"] == (1, 2) and module.LEG_TYPES["mt"] == (2, 0)  # CROWN: em = electron leg 1, muon leg 2
