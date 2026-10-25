"""The input validation of the NN phase (nn_limit_spec.md 7.1, NV1-NV3; switches nn_validation and pseudo_shifts).

Inputs: the union of the NN inputs read from n-tuple columns by the planned models (NN_INPUTS with the models that read
them): the thesis recipe (R3-R5, profile thesis6) and M0 (m0_v7b5) read the 20 thesis inputs, of which mtt_coll_approx
is computed in the graph from the legs and the MET; M1 reads the same 20; the step-2 models M2 and M3 (and the XGBoost
reference) the token fields of the eight object slots and the 25 HL features (bbtautau-nn schema.py). Not inputs: the
generator truth and the raw vsJet scores (iso_1 in tt). Constant columns are no test: mass_1 in mt (the muon mass) and
tau_decaymode_1 in et and mt (-1).

Variables (switch nn_validation): every input of a channel as a control variable in fine bins (VALIDATION_BINNING,
the GoF merges them to equal prediction, shapesmith.input_gof), the value limited to the binned range so that the
under- and overflow land in the edge bins (pandas.eval has no clip: x (x >= a) + a (x < a)); integer variables one bin
per value; mtt_coll_approx as the expression of the old CROWN formula (COLLINEAR_MASS); the b-tag scores in 1000 bins on
[0, 1] (NV2 resolves the working-point bins of BTAG_WP in them); the eta-phi occupancy of the legs and jets (D6) as the
index of an eta x phi cell (OCCUPANCY).

b-tag working points (NV2): BTAG_WP, the M and T thresholds of UParTAK4B per era from the BTV payload that CROWN pins
(bbtautau btag_payloads.BTV_UPART_PAYLOADS: /cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run2-<era>-UL-NanoAODv15/
2026-06-18/btagging.json.gz, correction UParTAK4_wp_values); CROWN's n_bjets counts the jets above M.

Pseudo-shifts (switch pseudo_shifts, NV1 with NV9 size): PRELIMINARY shape nuisances for the jet, MET and FastMTT inputs
until the shift production P6, as column variations derived from the nominal columns:
- CMS_pseudo_jes_<era>: the four momenta of the b pair and the two leading jets x (1 +- 0.03) (pt and mass; the dijet
  masses and pT scale exactly), on MC;
- CMS_pseudo_tes_<era> (MC) and CMS_pseudo_emb_tes_<era> (embedding): the tau_h legs x (1 +- 0.015);
- the MET by propagation: minus the change of the transverse momenta of the shifted objects (the b pair and the leading
  jets counted once: a leading jet equal to a b-pair jet is the same jet);
- recomputed from the shifted objects and MET with the CROWN definitions: m_vis, pt_vis, mt_1, mt_2, mt_tot,
  pzetamissvis (zeta = the bisector of the legs, alpha = 0.85), mass_tautaubb and pt_tautaubb (legs + b pair + MET with
  eta = 0 and mass 0), mtt_coll_approx (an expression of the columns), every cut (e.g. mt_1 < 70, the tau pT thresholds).
Approximations: the jets beyond the two leading jets and the b pair are not propagated to the MET; n_jets, n_bjets and
metSumEt stay (no jets below the thresholds in the skims; CROWN takes metSumEt from NanoAOD unshifted); objects below the
CROWN thresholds are not in the n-tuples, so an Up shift misses the events that would enter there (the datacards
symmetrise the pairs, Analysis.symmetrised_variations); FastMTT is not refitted: m_fastmtt and pt_fastmtt follow the mean
response of a refit (FASTMTT_*, measured with the standalone FastMTT of CROWN on 2018 events; per event a refit moves by
1-6 % more, the jitter of its likelihood minimum, which a scale cannot reproduce), eta_fastmtt stays; the tau_h shift
applies to every tau_h candidate (also jet and lepton fakes).
"""
from __future__ import annotations

import math

import numpy as np
from shapesmith.model import ColumnVariation, Variable

from bbtautau_shapesmith.ml import KINEMATICS, OBJECTS, RESKIM, THESIS_INPUTS

COLLINEAR_INPUTS = ("pt_1", "eta_1", "phi_1", "mass_1", "pt_2", "eta_2", "phi_2", "mass_2", "met", "metphi")
STEP2_HL = THESIS_INPUTS + ("mtt_coll_approx", "mt_1", "mt_2", "pzetamissvis", "mass_tautaubb", "pt_tautaubb")
COMPUTED = ("mtt_coll_approx",)
# the models that read an input (from the column itself or, for mtt_coll_approx, through it)
MODELS = ("R3-R5", "M0", "M1", "M2/M3")


def _uses(name: str) -> tuple[str, ...]:
    thesis = name in THESIS_INPUTS or name in COMPUTED
    collinear = name in COLLINEAR_INPUTS
    step2 = name in OBJECTS or name in RESKIM or name in KINEMATICS or name in STEP2_HL
    return tuple(model for model, used in zip(MODELS, (thesis or collinear, thesis or collinear, thesis or collinear, step2)) if used)


NN_INPUTS = {name: _uses(name) for name in dict.fromkeys(THESIS_INPUTS + COMPUTED + COLLINEAR_INPUTS + OBJECTS + RESKIM + KINEMATICS + STEP2_HL)}
# no input in a channel: the raw vsJet score of the leading tau (tt), constants (the muon mass, no decay mode)
NOT_INPUTS = {"tt": ("iso_1",), "mt": ("mass_1", "tau_decaymode_1"), "et": ("tau_decaymode_1",)}

# fine bins (low, high, bins) or integer ranges ("int", first, last)
_PT = (0.0, 600.0, 300)
_ETA = (-2.5, 2.5, 100)
_PHI = (-math.pi, math.pi, 64)
# the b-tag scores: 0.001 up to 0.99, 0.0001 above (b jets pile up near 1), and the M and T thresholds of every era as
# edges, so that the working-point bins of NV2 are exact unions of fine bins
_SCORE = ("edges",)
VALIDATION_BINNING = {
    "pt_1": _PT, "eta_1": _ETA, "phi_1": _PHI, "mass_1": (-0.5, 2.5, 150), "iso_1": (0.0, 0.5, 100), "tau_decaymode_1": ("int", 0, 11),
    "pt_2": _PT, "eta_2": _ETA, "phi_2": _PHI, "mass_2": (0.0, 2.5, 125), "tau_decaymode_2": ("int", 0, 11),
    "bpair_pt_1": (0.0, 800.0, 400), "bpair_eta_1": _ETA, "bpair_phi_1": _PHI, "bpair_mass_1": (0.0, 100.0, 200), "bpair_btag_value_1": _SCORE,
    "bpair_pt_2": (0.0, 800.0, 400), "bpair_eta_2": _ETA, "bpair_phi_2": _PHI, "bpair_mass_2": (0.0, 100.0, 200), "bpair_btag_value_2": _SCORE,
    "jpt_1": (0.0, 800.0, 400), "jeta_1": _ETA, "jphi_1": _PHI, "jtag_value_1": _SCORE,
    "jpt_2": (0.0, 800.0, 400), "jeta_2": _ETA, "jphi_2": _PHI, "jtag_value_2": _SCORE,
    "met": (0.0, 600.0, 300), "metphi": _PHI, "metSumEt": (0.0, 3000.0, 300),
    "m_fastmtt": (0.0, 1500.0, 300), "pt_fastmtt": (0.0, 800.0, 400), "eta_fastmtt": (-6.0, 6.0, 120), "phi_fastmtt": _PHI,
    "mt_1": (0.0, 300.0, 300), "mt_2": (0.0, 500.0, 250), "mt_tot": (0.0, 1000.0, 250), "pzetamissvis": (-500.0, 300.0, 400),
    "mass_tautaubb": (0.0, 3000.0, 300), "pt_tautaubb": (0.0, 800.0, 400), "deltaR_ditaupair": (0.0, 6.0, 120), "bpair_deltaR": (0.0, 6.0, 120),
    "bpair_m_inv": (0.0, 1500.0, 300), "bpair_pt_dijet": (0.0, 800.0, 400), "mjj": (0.0, 2000.0, 400), "pt_dijet": (0.0, 800.0, 400),
    "m_vis": (0.0, 800.0, 400), "pt_vis": (0.0, 600.0, 300), "n_jets": ("int", 0, 12), "n_bjets": ("int", 0, 6),
    "mtt_coll_approx": (0.0, 1500.0, 300),
}
# the collinear mass of the old CROWN (quantities.cxx:675-725, bbtautau-nn schema.mtt_coll_approx): x_1 = met/pt_1
# sin(phi_2 - phi_met)/sin(phi_2 - phi_1), x_2 = met/pt_2 sin(phi_1 - phi_met)/sin(phi_1 - phi_2), the mass of
# (1 + x_1) p_1 + (1 + x_2) p_2 (a negative factor flips the momentum and keeps the energy positive); -10 for
# |sin(phi_2 - phi_1)| < 1e-6 (the denominator kept finite there), pt and met > 0 in every selected event
_SIN = "sin(phi_2 - phi_1)"
_GUARD = f"(abs({_SIN}) < 1e-6)"
_DENOMINATOR = f"({_SIN} + {_GUARD} * 1.0)"
_A1 = f"(1 + met / pt_1 * sin(phi_2 - metphi) / {_DENOMINATOR})"
_A2 = f"(1 - met / pt_2 * sin(phi_1 - metphi) / {_DENOMINATOR})"
_PX = f"({_A1} * pt_1 * cos(phi_1) + {_A2} * pt_2 * cos(phi_2))"
_PY = f"({_A1} * pt_1 * sin(phi_1) + {_A2} * pt_2 * sin(phi_2))"
_PZ = f"({_A1} * pt_1 * sinh(eta_1) + {_A2} * pt_2 * sinh(eta_2))"
_E = f"(abs({_A1}) * sqrt((pt_1 * cosh(eta_1)) ** 2 + mass_1 ** 2) + abs({_A2}) * sqrt((pt_2 * cosh(eta_2)) ** 2 + mass_2 ** 2))"
_M2 = f"({_E} ** 2 - {_PX} ** 2 - {_PY} ** 2 - {_PZ} ** 2)"
COLLINEAR_MASS = f"((1.0 - {_GUARD} * 1.0) * sqrt(({_M2} + abs({_M2})) / 2) - 10.0 * {_GUARD} * 1.0)"
EXPRESSIONS = {"mtt_coll_approx": COLLINEAR_MASS}

# D6: the eta x phi cell of an object, index i_eta * N_PHI + i_phi (|eta| < 2.5 in 20 cells, phi in 24)
OCCUPANCY_ETA, OCCUPANCY_N_ETA, OCCUPANCY_N_PHI = 2.5, 20, 24
OCCUPANCY = {"leg_1": ("eta_1", "phi_1"), "leg_2": ("eta_2", "phi_2"), "jet_1": ("jeta_1", "jphi_1"), "jet_2": ("jeta_2", "jphi_2"),
             "bjet_1": ("bpair_eta_1", "bpair_phi_1"), "bjet_2": ("bpair_eta_2", "bpair_phi_2")}

# NV2: the M and T thresholds of UParTAK4B per era (module docstring), WP bins < M, M-T, >= T
BTAG_WP = {"2016preVFP": (0.1847, 0.5467), "2016postVFP": (0.1898, 0.5538), "2017": (0.1776, 0.5755), "2018": (0.161, 0.5405)}
BTAG_SCORES = ("bpair_btag_value_1", "bpair_btag_value_2", "jtag_value_1", "jtag_value_2")
# the inputs of FastMTT (CROWN quantities::FastMtt), kept in the validation skims for a refit
FASTMTT_INPUTS = ("pt_1", "eta_1", "phi_1", "mass_1", "pt_2", "eta_2", "phi_2", "mass_2", "met", "metphi", "metcov00", "metcov01", "metcov11")

# pseudo-shifts (module docstring): the relative size of the jet and the tau_h energy shifts
PSEUDO_JES, PSEUDO_TES = 0.03, 0.015
PSEUDO_JES_NAME, PSEUDO_TES_NAME, PSEUDO_EMB_TES_NAME = "CMS_pseudo_jes", "CMS_pseudo_tes", "CMS_pseudo_emb_tes"
PZETA_ALPHA = 0.85  # CROWN quantities::PzetaMissVis
# the mean response of a FastMTT refit (the standalone build of CROWN core src/SVFit, bitwise equal to the friend) to the
# pseudo-shifts, 20 000 events per channel of 2018 (n_bjets >= 1, opposite sign): tau_h +-1.5 % moves the quartiles of
# m_fastmtt by +-0.5-0.6 % in et/mt and +-1.0 % in tt, 0.7 (0.69-0.73) of the change of m_vis, and pt_fastmtt by
# +-0.2-0.35 % per shifted leg; jets +-3 % move m_fastmtt by +-0.6-0.9 % (0.25 of the jet shift) and pt_fastmtt by about
# 0.6 of the MET change of the jets (+-1.4-1.6 % against +-2.3-2.6 % with all of it)
FASTMTT_TES_MASS_POWER = 0.7  # m_fastmtt x (m_vis' / m_vis) ** 0.7
FASTMTT_TES_PT = 0.22  # pt_fastmtt x (1 + 0.22 sum of the leg shifts)
FASTMTT_JES_MASS = 0.25  # m_fastmtt x (1 + 0.25 jet shift)
FASTMTT_MET_FRACTION = 0.6  # pt, phi_fastmtt: the vector plus 0.6 of the MET change of the jets


def clipped(expr: str, low: float, high: float) -> str:
    """`expr` limited to [low, high] with a margin of 1e-6 of the range inside, so that it lands in the edge bins:
    x (x >= a) + a (x < a), exact in float32 (unlike (x + a + |x - a|) / 2, which loses the bound for |x| >> |a|); NaN
    stays NaN."""
    margin = (high - low) * 1e-6
    low, high = low + margin, high - margin
    lower = f"(({expr}) * (({expr}) >= {low!r}) + {low!r} * (({expr}) < {low!r}))"
    return f"({lower} * ({lower} <= {high!r}) + {high!r} * ({lower} > {high!r}))"


def score_edges() -> tuple[float, ...]:
    """The fine edges of a b-tag score (_SCORE)."""
    grid = np.concatenate([np.arange(0, 990) / 1000.0, 0.99 + np.arange(0, 101) / 10000.0])
    thresholds = [threshold for points in BTAG_WP.values() for threshold in points]
    return tuple(sorted({float(round(x, 9)) for x in grid} | set(thresholds)))


def _edges(spec) -> tuple[float, ...]:
    if spec[0] == "int":
        return tuple(float(x) for x in np.arange(spec[1] - 0.5, spec[2] + 1.0, 1.0))
    if spec[0] == "edges":
        return score_edges()
    low, high, n = spec
    return tuple(float(round(x, 9)) for x in np.linspace(low, high, n + 1))


def inputs(channel: str) -> tuple[str, ...]:
    """The NN inputs tested in a channel (module docstring)."""
    return tuple(name for name in NN_INPUTS if name not in NOT_INPUTS.get(channel, ()))


def occupancy_expression(eta: str, phi: str) -> str:
    """The index of the eta x phi cell of (eta, phi); an object outside |eta| < 2.5 in the edge cells, an absent one
    (the CROWN default -10, e.g. no b pair with fewer than two central jets) at -1000, outside the histogram."""
    eta_cell = f"floor(({clipped(eta, -OCCUPANCY_ETA, OCCUPANCY_ETA)} + {OCCUPANCY_ETA!r}) / {2 * OCCUPANCY_ETA / OCCUPANCY_N_ETA!r})"
    phi_cell = f"floor(({clipped(phi, -math.pi, math.pi)} + {math.pi!r}) / {2 * math.pi / OCCUPANCY_N_PHI!r})"
    return f"(({eta_cell} * {OCCUPANCY_N_PHI} + {phi_cell}) * ({eta} > -9) - 1000 * ({eta} <= -9))"


def validation_variables(channel: str, fastmtt: bool = True) -> dict[str, Variable]:
    """The control variables of the switch nn_validation: every NN input of the channel (the FastMTT ones with its
    friend), `yield` and the occupancy cells occupancy_<object>."""
    names = [name for name in inputs(channel) if fastmtt or "fastmtt" not in name]
    variables = {name: Variable(name, clipped(EXPRESSIONS.get(name, name), *_range(name)), _edges(VALIDATION_BINNING[name])) for name in names}
    cells = tuple(float(x) for x in np.arange(-0.5, OCCUPANCY_N_ETA * OCCUPANCY_N_PHI, 1.0))
    variables.update({f"occupancy_{name}": Variable(f"occupancy_{name}", occupancy_expression(*columns), cells) for name, columns in OCCUPANCY.items()})
    variables["yield"] = Variable("yield", "1.0", (0.5, 1.5))
    return variables


def _range(name: str) -> tuple[float, float]:
    edges = _edges(VALIDATION_BINNING[name])
    return edges[0], edges[-1]


def dilepton_validation_variables() -> dict[str, Variable]:
    """The variables of NV2 in em, mm and ee: the four b-tag scores, the jet kinematics and the occupancy of the jets."""
    names = BTAG_SCORES + ("bpair_pt_1", "bpair_pt_2", "bpair_eta_1", "bpair_eta_2", "jpt_1", "jpt_2", "jeta_1", "jeta_2", "n_jets", "n_bjets", "met")
    variables = {name: Variable(name, clipped(name, *_range(name)), _edges(VALIDATION_BINNING[name])) for name in names}
    cells = tuple(float(x) for x in np.arange(-0.5, OCCUPANCY_N_ETA * OCCUPANCY_N_PHI, 1.0))
    variables.update({f"occupancy_{name}": Variable(f"occupancy_{name}", occupancy_expression(*OCCUPANCY[name]), cells) for name in ("jet_1", "jet_2", "bjet_1", "bjet_2")})
    variables["yield"] = Variable("yield", "1.0", (0.5, 1.5))
    return variables


# -- pseudo-shifts ---------------------------------------------------------------------------------------------------


def _scaled(column: str, factor: float) -> str:
    return column if factor == 1.0 else f"({factor!r} * {column})"


def _jet_terms() -> list[tuple[str, str, str]]:
    """(pt, phi, weight) of the distinct jets among the b pair and the two leading jets: a leading jet with the pt and
    eta of a b-pair jet (to 1e-3) is that jet; absent objects (pt -10) count 0."""
    def same(jet: int, b: int) -> str:
        return f"((abs(jpt_{jet} - bpair_pt_{b}) < 0.001 * jpt_{jet}) & (abs(jeta_{jet} - bpair_eta_{b}) < 0.001))"

    terms = [(f"bpair_pt_{b}", f"bpair_phi_{b}", f"(bpair_pt_{b} > 0)") for b in (1, 2)]
    terms += [(f"jpt_{j}", f"jphi_{j}", f"((jpt_{j} > 0) & ~({same(j, 1)} | {same(j, 2)}))") for j in (1, 2)]
    return terms


def _momentum_sum(terms: list[tuple[str, str, str, float]], function: str) -> str:
    """The sum of delta * weight * pt * function(phi) over the shifted objects (pt, phi, weight, delta), the deltas
    factored out; 0 without any."""
    deltas: dict[float, list[str]] = {}
    for pt, phi, weight, delta in terms:
        deltas.setdefault(delta, []).append(f"{pt} * {function}({phi})" if weight == "1.0" else f"{weight} * {pt} * {function}({phi})")
    return " + ".join(f"{delta!r} * ({' + '.join(parts)})" for delta, parts in deltas.items()) or "0.0"


def pseudo_shift(legs: dict[int, float], jets: float = 1.0) -> dict[str, str]:
    """The derived columns (ColumnVariation.derived) of the legs scaled by `legs` (leg -> factor) and the jets by
    `jets`, with the MET propagated and the derived quantities recomputed (module docstring)."""
    f = {leg: legs.get(leg, 1.0) for leg in (1, 2)}
    derived = {}
    for leg, factor in f.items():
        if factor != 1.0:
            derived[f"pt_{leg}"] = _scaled(f"pt_{leg}", factor)
            derived[f"mass_{leg}"] = _scaled(f"mass_{leg}", factor)
    leg_terms = [(f"pt_{leg}", f"phi_{leg}", "1.0", factor - 1.0) for leg, factor in f.items() if factor != 1.0]
    jet_terms = [(pt, phi, weight, jets - 1.0) for pt, phi, weight in _jet_terms()] if jets != 1.0 else []
    b = {i: (_scaled(f"bpair_pt_{i}", jets), _scaled(f"bpair_mass_{i}", jets)) for i in (1, 2)}
    if jets != 1.0:
        for column in ("bpair_pt_1", "bpair_pt_2", "bpair_mass_1", "bpair_mass_2", "jpt_1", "jpt_2", "bpair_m_inv", "bpair_pt_dijet", "mjj", "pt_dijet"):
            derived[column] = _scaled(column, jets)
    # the MET: minus the change of the transverse momenta of the shifted objects
    terms = leg_terms + jet_terms
    mx, my = f"(met * cos(metphi) - ({_momentum_sum(terms, 'cos')}))", f"(met * sin(metphi) - ({_momentum_sum(terms, 'sin')}))"
    met = f"sqrt({mx} ** 2 + {my} ** 2)"
    derived["met"], derived["metphi"] = met, f"arctan2({my}, {mx})"
    pt = {leg: _scaled(f"pt_{leg}", f[leg]) for leg in (1, 2)}
    mass = {leg: _scaled(f"mass_{leg}", f[leg]) for leg in (1, 2)}

    def transverse_mass(leg: int) -> str:
        # 2 (pt |MET| - p_T . MET), >= 0
        square = f"2 * ({pt[leg]} * {met} - {pt[leg]} * (cos(phi_{leg}) * {mx} + sin(phi_{leg}) * {my}))"
        return f"sqrt(({square} + abs({square})) / 2)"

    derived["mt_1"], derived["mt_2"] = transverse_mass(1), transverse_mass(2)
    mixed = f"2 * {pt[1]} * {pt[2]} * (1 - cos(phi_1 - phi_2))"
    derived["mt_tot"] = f"sqrt({derived['mt_1']} ** 2 + {derived['mt_2']} ** 2 + {mixed})"
    # MET . zeta - alpha pzeta_vis, zeta the unit bisector of the legs: u_i . (u_1 + u_2) = 1 + cos(phi_1 - phi_2)
    bisector = "sqrt((cos(phi_1) + cos(phi_2)) ** 2 + (sin(phi_1) + sin(phi_2)) ** 2)"
    derived["pzetamissvis"] = f"(({mx} * (cos(phi_1) + cos(phi_2)) + {my} * (sin(phi_1) + sin(phi_2)) - {PZETA_ALPHA!r} * ({pt[1]} + {pt[2]}) * (1 + cos(phi_1 - phi_2))) / {bisector})"
    if leg_terms:
        # exact from the nominal: E_1 E_2 - p_1 . p_2 = (m_vis^2 - m_1^2 - m_2^2) / 2, scaled by f_1 f_2
        derived["m_vis"] = f"sqrt(abs({mass[1]} ** 2 + {mass[2]} ** 2 + {f[1] * f[2]!r} * (m_vis ** 2 - mass_1 ** 2 - mass_2 ** 2)))"
        derived["pt_vis"] = f"sqrt(abs({pt[1]} ** 2 + {pt[2]} ** 2 + 2 * {pt[1]} * {pt[2]} * cos(phi_1 - phi_2)))"
    # legs + b pair + MET (eta 0, mass 0)
    objects = [(pt[1], "eta_1", "phi_1", mass[1]), (pt[2], "eta_2", "phi_2", mass[2]), (b[1][0], "bpair_eta_1", "bpair_phi_1", b[1][1]), (b[2][0], "bpair_eta_2", "bpair_phi_2", b[2][1])]
    px = " + ".join(f"{p} * cos({phi})" for p, _, phi, _ in objects) + f" + {mx}"
    py = " + ".join(f"{p} * sin({phi})" for p, _, phi, _ in objects) + f" + {my}"
    pz = " + ".join(f"{p} * sinh({eta})" for p, eta, _, _ in objects)
    energy = " + ".join(f"sqrt(({p} * cosh({eta})) ** 2 + {m} ** 2)" for p, eta, _, m in objects) + f" + {met}"
    square = f"(({energy}) ** 2 - ({px}) ** 2 - ({py}) ** 2 - ({pz}) ** 2)"
    derived["mass_tautaubb"] = f"sqrt(({square} + abs({square})) / 2)"
    derived["pt_tautaubb"] = f"sqrt(({px}) ** 2 + ({py}) ** 2)"
    # FastMTT, not refitted (module docstring): the mean response of a refit as scale factors and a fraction of the MET
    # change of the jets
    tes = sum(factor - 1.0 for factor in f.values())
    mass_scale = f"(({derived['m_vis']}) / m_vis) ** {FASTMTT_TES_MASS_POWER!r}" if leg_terms else "1.0"
    if jets != 1.0:
        mass_scale = f"{1.0 + FASTMTT_JES_MASS * (jets - 1.0)!r}"
    if leg_terms or jets != 1.0:
        derived["m_fastmtt"] = f"(m_fastmtt * {mass_scale})"
    if leg_terms:
        derived["pt_fastmtt"] = f"({1.0 + FASTMTT_TES_PT * tes!r} * pt_fastmtt)"
    if jet_terms:
        fx = f"(pt_fastmtt * cos(phi_fastmtt) - {FASTMTT_MET_FRACTION!r} * ({_momentum_sum(jet_terms, 'cos')}))"
        fy = f"(pt_fastmtt * sin(phi_fastmtt) - {FASTMTT_MET_FRACTION!r} * ({_momentum_sum(jet_terms, 'sin')}))"
        derived["pt_fastmtt"], derived["phi_fastmtt"] = f"sqrt({fx} ** 2 + {fy} ** 2)", f"arctan2({fy}, {fx})"
    return derived


def tau_legs(channel: str) -> tuple[int, ...]:
    return (1, 2) if channel == "tt" else (2,)


def pseudo_shift_variations(channel: str, era: str, embedded: bool) -> tuple[ColumnVariation, ...]:
    """The preliminary pseudo-shifts of a tau channel (module docstring): the jets on MC, the tau_h legs on MC and, in an
    embedded channel, on the embedding (its own nuisance)."""
    result = ()
    for direction, sign in (("Up", 1.0), ("Down", -1.0)):
        jets = pseudo_shift({}, 1.0 + sign * PSEUDO_JES)
        taus = pseudo_shift({leg: 1.0 + sign * PSEUDO_TES for leg in tau_legs(channel)})
        result += (ColumnVariation(f"{PSEUDO_JES_NAME}_{era}{direction}", derived=jets, applies_to=("mc",)),
                   ColumnVariation(f"{PSEUDO_TES_NAME}_{era}{direction}", derived=taus, applies_to=("mc",)))
        if embedded:
            result += (ColumnVariation(f"{PSEUDO_EMB_TES_NAME}_{era}{direction}", derived=taus, applies_to=("embedding",)),)
    return result


def pseudo_shift_names(era: str, embedded: bool, enabled: bool = True) -> tuple[str, ...]:
    """The pseudo-shift nuisances (without the direction) of a run with the switch pseudo_shifts (`enabled`), for
    Analysis.symmetrised_variations; `embedded`: a channel of the run takes its genuine taus from the embedding."""
    if not enabled:
        return ()
    return (f"{PSEUDO_JES_NAME}_{era}", f"{PSEUDO_TES_NAME}_{era}") + ((f"{PSEUDO_EMB_TES_NAME}_{era}",) if embedded else ())
