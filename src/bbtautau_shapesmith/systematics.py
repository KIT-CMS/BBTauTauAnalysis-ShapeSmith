"""Systematic uncertainties per era: b-tag and prefiring weight variations, the CROWN shifts of the embedded sample and
of the fake-factor friend (column variations), normalisation uncertainties (lnN). The era tokens follow the
correlation scheme of correlations.py (v7 spec 6.7); in v7 (nominal only, E34) the datacards carry the lnN and the
variations of the nominal trees (b tag, fake factors, prefiring). The switch study_uncertainties adds the uncertainties
sized in the v7b studies (study_uncertainties.py): the embedding tau ID as exact weight factors and its ES as lnN, the
POG MC tau ES, vsJet components and di-tau trigger SF as lnN, the residual decay-mode non-closure of the fake factors as
a shape, the MET residual, the HIPM lepton efficiency, the 2016 halves of the fake factors and K4 as lnN. The stages of
the NN phase (nn_limit_spec.md 8) add weight variations by switch: S0+ the top-pT shape of ttbar (ttbar_shape, SH7) and the
heavy-flavour normalisation at >= 3 b tags (heavy_flavour, SH8), S1 the signal theory shapes normalised to the inclusive
rate (signal_theory, SH14; with the shift production).

The shift production v8 (switch shift_set v8; v8 spec 6.1, SH16): crown_shifts_v8 lists every CROWN shift of the
simulation and the embedding of a channel with the datacard nuisance made of it (the table of the spec), v8_variations
declares them and v8_variation_sums builds the nuisances that are sums of shifts (the JER of its six regions, the
embedding tau ES with its partial correlation to the simulation, V20). The CROWN names are those of crown_names.py. With
them the stand-ins of the study switch vanish where a shift takes their place (study_lnn); the weight shape of the
tau_h above 140 GeV (tau_highpt_variations, V35), the parton shower of ttbar (ttbar_theory_variations, Q6) and the PDF of
the signal as the quadrature sum of its eigenvectors (V27) come with the production as well.
"""
from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, replace

from shapesmith.histogram import part_of
from shapesmith.model import NOMINAL, AnalysisError, ColumnVariation, LnN, Process, QuadratureSum, Sample, Selection, VariationSum, WeightVariation

from bbtautau_shapesmith import crown_names as crown
from bbtautau_shapesmith import study_uncertainties as study_sizes
from bbtautau_shapesmith.constants import (
    BTAG_COMPONENTS, DY_TABLE_LNN, EMBEDDING_TAU_DECAY_MODES, FF_COLUMNS, FF_SHIFTS_LT, FF_SHIFTS_TT, FF_SHIFTS_W, LT_CHANNELS, PREFIRING_NUISANCE, PREFIRING_VARIATIONS,
    RECOIL_ERAS, TAU_CHANNELS, TAU_LEGS, TAU_VS_ELE_WP, TAU_VS_JET_WP, YEAR,
)
from bbtautau_shapesmith.correlations import nuisance
from bbtautau_shapesmith.processes import EMBEDDED, SIGNAL, VH_TAUTAU
from bbtautau_shapesmith.theory_norms import RUNS_NORM
from bbtautau_shapesmith.weights import TOP_PT

DIRECTIONS = ("Up", "Down")
# LUM POG Cholesky scheme for 2016+2017+2018 (LumiRecommendationsRun2 r13, 2026-01-14; spec 6.6, E20): nuisance -> the
# lnN of each year; both 2016 halves take the 2016 column. MC only (not the embedded sample, not the jet fakes).
LUMI = {
    "lumi_13TeV_1516_l": {"2016": 1.0118},
    "lumi_13TeV_151617_l": {"2016": 1.0004, "2017": 1.0055},
    "lumi_13TeV_15161718_l": {"2016": 1.0035, "2017": 1.0061, "2018": 1.0084},
}
LEPTON_ID = 1.02  # e/mu ID and isolation on MC (the correlated part, K4)
# E27: the embedded sample shares half of the MC lepton-ID nuisance (0.5 x 2 %) and has its own part (0.866 x 2 %)
EMBEDDING_SHARED, EMBEDDING_OWN = 0.5, 0.866
TRIGGER_LNN = 1.02  # the single-lepton trigger SFs of the nominal production have no variations: +-2 % (TauAnalysis convention, spec C6)
BORROWED_LNN = 1.05  # a process of samples borrowed from another era (3.3): the detector differences
# NN spec 8, S0+ (SH7; v7_analyse 4.1, 11.2 no. 2): the top-pT shape of ttbar as in the tau-ID measurement (K9, TOP
# recommendation), the difference with (the nominal) against without the reweighting, symmetrised: Down without the
# weight, Up its mirror 2 w - 1 around the nominal; on every process that carries the top-pT weight (weights.TOP_PT: the
# TT parts of the tau channels, TT of em), correlated over the channels and eras
TTBAR_SHAPE = "CMS_htt_ttbarShape"
# S0+ (SH8, N12; v7_analyse 4.5, 11.2 no. 1): the heavy flavour the simulation lacks at >= 3 b tags (data/prediction of
# btag3p in every era 1.24-1.37 in em, 1.26-1.35 in mm + ee with the DY table, 0.99-1.29 in et and mt; the b-tag SF
# +1 sigma reaches only 1.25-1.35, v5): ttbb, Z/V + bb and g -> bb. A 30 % normalisation (lnN 1.30, Down 1 / 1.30) of the
# heavy-flavour component of the simulated backgrounds, the events with >= 3 b tags whose b pair is two heavy-flavour jets
# (hadron flavour b or c, columns of the re-skim X8). Written as a weight variation, so that the share of the component
# sets its size in each bin. The b pair of ttbar is that of the top decays below 3 b tags, and the DY table corrects DY in
# btag1 and btag2 (measured with exactly 2 b tags, CMS_htt_dy_table): the component starts at 3 b tags.
HEAVY_FLAVOUR = "CMS_htt_heavy_flavour"
HEAVY_FLAVOUR_LNN = 1.30
HEAVY_FLAVOUR_COMPONENT = "((n_bjets >= 3) & (bpair_hadronFlavour_1 >= 4) & (bpair_hadronFlavour_2 >= 4))"
HEAVY_FLAVOUR_WEIGHT = {"heavy_flavour": "1.0"}  # the weight it replaces, on the simulated backgrounds of a run with it
# S1 (SH14, v8 spec SH20): the theory shapes of a process, each normalised to the inclusive rate of its samples, so
# that only acceptance and shape remain (for the signal the inclusive cross section is QCDscale_HH and PDF_alphas_HH):
# kind of the switch -> datacard name -> the CROWN shift whose weight it reads. The QCD scales are the shifts
# QCDscale_ren and QCDscale_fac of lhe_scale_weight (LHEScaleWeight: muR and muF times 2 Up, 0.5 Down), the parton
# shower the shifts ps_isr and ps_fsr (PSWeight: the ISR and FSR scale times 2 Up, 0.5 Down, so Up is less radiation);
# the PDF is the quadrature sum per bin of the 100 eigenvector weights of the nominal tree (V27), `pdf` below.
THEORY_COLUMNS = {**dict.fromkeys(crown.QCD_SCALES, crown.LHE_SCALE_WEIGHT), **dict.fromkeys(crown.PARTON_SHOWER, crown.PS_WEIGHT)}  # shift -> the weight it moves
PDF = "pdf"
SIGNAL_THEORY = {
    "qcd_scale": {"QCDscale_ren_HH_acc": "QCDscale_ren", "QCDscale_fac_HH_acc": "QCDscale_fac"},
    "parton_shower": {"ps_isr_HH": "ps_isr", "ps_fsr_HH": "ps_fsr"},
    "pdf": {"pdf_HH_acc": PDF},
}
SIGNAL_THEORY_WEIGHT = {"theory": "1.0"}  # the weight they replace, on the signal of a run with them
# Q6, P8 (TOP recommendation): the parton shower of ttbar, named per process (not correlated with the signal or single
# top), correlated between the eras, normalised before the acceptance; no KIT-legacy precedent, released by the user
TTBAR_THEORY = {"ps_isr_ttbar": "ps_isr", "ps_fsr_ttbar": "ps_fsr"}
TTBAR_THEORY_WEIGHT = {"ttbar_theory": "1.0"}  # the weight they replace, on the ttbar processes of a run with them
TTBAR_GROUP = "TT"
# V35 (SH28): genuine tau_h above 140 GeV, where the POG recommends the pT-binned vsJet SF while the nominal stays the
# DM-binned one. A weight shape per era, uncorrelated between the eras, no lnN; per leg above the threshold the envelope
# of the difference of the two sets and the uncertainty of the pT-binned one: Up max(SF_pt_up, SF_pt, SF_dm) / SF_dm,
# Down min(SF_pt_down, SF_pt, SF_dm) / SF_dm; 1 below it and for every other tau candidate. No precedent, released.
TAU_HIGHPT = "CMS_eff_t_highpT"
TAU_ID_WEIGHT = "tau_id"  # the weight of the simulated processes it replaces (weights.mc_weights)
# the shift model of the v8 production (crown_shifts_v8): which samples of the kinds carry a shift
ALL, RECOIL, NO_RECOIL = "all", "recoil", "no_recoil"
RECOIL_GROUPS = ("DY", "W")  # the recoil-corrected samples of the recoil eras (CROWN sample types dyjets and wjets, V9)
MC, EMB, MC_AND_EMB = ("mc",), ("embedding",), ("mc", "embedding")


def btag_variations(era: str) -> tuple[WeightVariation, ...]:
    """Up/Down of the fixed-WP b-tag weight (C14): b/c and light jets, each correlated between the eras (no token) and
    uncorrelated (per era, the 2016 halves separate); spec 6.5."""
    result = ()
    for component in BTAG_COMPONENTS:
        flavour, correlation = component.split("_")
        name = nuisance(f"CMS_btag_fixedWP_{flavour}_{correlation}", f"btag_{correlation}", era)
        result += tuple(WeightVariation(f"{name}{d}", {"btag": f"btag_weight_upart_{d.lower()}_{component}"}) for d in DIRECTIONS)
    return result


def prefiring_variations() -> tuple[WeightVariation, ...]:
    """The prefiring weight Up/Down of the nominal tree (C11); one CMS_prefiring for 2016-2018 (E11), written as lnN
    (Analysis.lnn_variations)."""
    return tuple(WeightVariation(f"{PREFIRING_NUISANCE}{d}", {"prefiring": column}) for d, column in PREFIRING_VARIATIONS.items())


def _shifts(name: str, suffix: str, applies_to: tuple[str, ...]) -> tuple[ColumnVariation, ...]:
    """The Up/Down pair of one CROWN shift: datacard name `name`Up/Down, branches `c + suffix`Up/Down."""
    return tuple(ColumnVariation(f"{name}{shift}", f"{suffix}{shift}", applies_to=applies_to) for shift in DIRECTIONS)


def _embedding_nuisances(channel: str, era: str) -> list[tuple[str, dict[int, str]]]:
    """(datacard name, decay mode -> CROWN shift) of the embedding tau corrections, one per measured category. The
    vsEle working point in the name decorrelates et (Tight) from mt and tt (VVLoose)."""
    wp = f"vsEle{TAU_VS_ELE_WP[channel]}"
    result = []
    for kind, crown in (("scale_t_emb", "scale_t_emb_DeepTau2018v2p5"), ("eff_t_emb", "eff_t_emb_DeepTau2018v2p5_VSjet")):
        for token, decay_modes in EMBEDDING_TAU_DECAY_MODES.items():
            shifts = {dm: f"CMS_{crown}_DM{dm}_{era}" for dm in decay_modes}
            result.append((nuisance(f"CMS_{kind}_dm{token}_{wp}", "embedding", era), shifts))
    return result


def embedding_variations(channel: str, era: str) -> tuple[ColumnVariation, ...]:
    """Tau ES and vsJet ID shifts of the embedded sample. A nuisance of one decay mode reads its CROWN shift; one of
    two decay modes reads each as a part of the sum of embedding_variation_sums."""
    result = ()
    for name, shifts in _embedding_nuisances(channel, era):
        if len(shifts) == 1:
            (shift,) = shifts.values()
            result += _shifts(name, f"__{shift}", ("embedding",))
            continue
        result += tuple(
            ColumnVariation(part_of(f"{name}{d}", f"dm{dm}"), f"__{shift}{d}", applies_to=("embedding",))
            for dm, shift in shifts.items()
            for d in DIRECTIONS
        )
    return result


def embedding_variation_sums(channel: str, era: str) -> tuple[VariationSum, ...]:
    """The nuisances of two decay modes (DM10 and DM11), each the sum of its per-decay-mode CROWN shifts: exact in et
    and mt (one tau per event), to first order in tt."""
    return tuple(VariationSum(name, tuple(f"dm{dm}" for dm in shifts)) for name, shifts in _embedding_nuisances(channel, era) if len(shifts) > 1)


def embedding_ttbar_nuisance(era: str) -> str:
    return nuisance("CMS_htt_emb_ttbar", "embedding", era)


def ff_variations(channel: str, era: str, wjets: bool = False) -> tuple[ColumnVariation, ...]:
    """The fake-factor friend shifts, per channel and year (2016 is measured on both halves); in tt a key shifts only
    the fake factor of its leg. With W+jets fake factors (ff_processes) et and mt add the W+jets keys."""
    keys = FF_SHIFTS_TT if channel == "tt" else FF_SHIFTS_LT + (FF_SHIFTS_W if wjets else ())
    return tuple(variation for key in keys for variation in _shifts(nuisance(f"CMS_ff_{key}_{channel}", "fake_factors", era), f"__{key}", ("data", "mc", "embedding")))


def with_weights(processes: Iterable[Process], names: Collection[str], weights: dict[str, str]) -> tuple[Process, ...]:
    """The processes, those in `names` with `weights` added after their own (the nominal factor that a weight
    variation replaces, e.g. HEAVY_FLAVOUR_WEIGHT)."""
    return tuple(replace(p, selection=Selection(p.selection.cuts, {**p.selection.weights, **weights})) if p.name in names else p for p in processes)


def ttbar_shape_variations(era: str) -> tuple[WeightVariation, ...]:
    """CMS_htt_ttbarShape (TTBAR_SHAPE): Down without the top-pT weight, Up its mirror 2 w - 1 around the nominal; in the
    nominal region, since the estimates take no weight variation."""
    name = nuisance(TTBAR_SHAPE, "theory", era)
    ((weight, column),) = TOP_PT.items()
    return (WeightVariation(f"{name}Up", {weight: f"(2.0 * {column} - 1.0)"}, regions=(NOMINAL,)), WeightVariation(f"{name}Down", {weight: "1.0"}, regions=(NOMINAL,)))


def heavy_flavour_variations(era: str) -> tuple[WeightVariation, ...]:
    """CMS_htt_heavy_flavour (HEAVY_FLAVOUR): the heavy-flavour component times 1.30 (Up) and 1 / 1.30 (Down) on the
    processes that carry HEAVY_FLAVOUR_WEIGHT; in the nominal region."""
    name = nuisance(HEAVY_FLAVOUR, "theory", era)
    ((weight, _),) = HEAVY_FLAVOUR_WEIGHT.items()
    shifts = (("Up", round(HEAVY_FLAVOUR_LNN - 1, 4)), ("Down", round(1 / HEAVY_FLAVOUR_LNN - 1, 4)))
    return tuple(WeightVariation(f"{name}{d}", {weight: f"(1 + {HEAVY_FLAVOUR_COMPONENT} * {shift!r})"}, regions=(NOMINAL,)) for d, shift in shifts)


def theory_norm(samples: Iterable[Sample], key: str) -> float | tuple[float, ...]:
    """The inclusive normalisation of a varied theory weight (theory_norms.RUNS_NORM: a CROWN shift with its
    direction, or `pdf`, the 100 eigenvectors) of the samples of a process: of one sample its own, of several the mean
    weighted with their cross sections, which keeps their summed inclusive rate."""
    samples = tuple(samples)
    missing = [sample.nick for sample in samples if key not in RUNS_NORM.get(sample.nick, {})]
    if missing or not samples:
        raise AnalysisError(f"no inclusive normalisation of {key} for {', '.join(missing) or 'a process without samples'} (theory_norms.RUNS_NORM; scripts/signal_theory_sums.py)")
    total = sum(sample.xsec for sample in samples)
    if key == PDF:
        return tuple(round(sum(sample.xsec * RUNS_NORM[sample.nick][key][member] for sample in samples) / total, 6) for member in range(len(crown.PDF_MEMBERS)))
    return round(sum(sample.xsec * RUNS_NORM[sample.nick][key] for sample in samples) / total, 6)


def _theory_variations(names: Mapping[str, str], weight: str, era: str, samples: tuple[Sample, ...]) -> tuple[WeightVariation, ...]:
    """Up and Down of each nuisance (datacard name -> CROWN shift) as the shifted weight over its inclusive
    normalisation, in place of `weight`; in the nominal region."""
    result = ()
    for name, shift in names.items():
        for direction in DIRECTIONS:
            varied = crown.shifted(THEORY_COLUMNS[shift], shift, direction)
            norm = theory_norm(samples, f"{shift}{direction}")
            result += (WeightVariation(f"{nuisance(name, 'theory', era)}{direction}", {weight: f"({varied} / {norm!r})"}, regions=(NOMINAL,)),)
    return result


def signal_theory_variations(era: str, kinds: Iterable[str], samples: Iterable[Sample]) -> tuple[WeightVariation, ...]:
    """The signal theory shapes of the kinds (SIGNAL_THEORY) on the process that carries SIGNAL_THEORY_WEIGHT, whose
    samples are `samples`: each direction the varied weight over its inclusive normalisation (theory_norm; the inclusive
    rate stays); for the PDF one template per eigenvector, the parts of signal_theory_sums; in the nominal region."""
    samples = tuple(samples)
    ((weight, _),) = SIGNAL_THEORY_WEIGHT.items()
    result = ()
    for kind in kinds:
        names = SIGNAL_THEORY[kind]
        result += _theory_variations({name: shift for name, shift in names.items() if shift != PDF}, weight, era, samples)
        for name in (name for name, shift in names.items() if shift == PDF):
            norms = theory_norm(samples, PDF)
            result += tuple(
                WeightVariation(part_of(nuisance(name, "theory", era), f"{member}"), {weight: f"({crown.pdf_weight(member)} / {norm!r})"}, regions=(NOMINAL,))
                for member, norm in zip(crown.PDF_MEMBERS, norms)
            )
    return result


def signal_theory_sums(era: str, kinds: Iterable[str]) -> tuple[QuadratureSum, ...]:
    """The PDF nuisance of the signal (V27): per bin the root of the summed squared differences of the 100 eigenvector
    templates to the nominal, each normalised to the inclusive rate (signal_theory_variations)."""
    names = [name for kind in kinds for name, shift in SIGNAL_THEORY[kind].items() if shift == PDF]
    return tuple(QuadratureSum(nuisance(name, "theory", era), tuple(f"{member}" for member in crown.PDF_MEMBERS)) for name in names)


def ttbar_theory_variations(era: str, samples: Iterable[Sample]) -> tuple[WeightVariation, ...]:
    """The parton shower of ttbar (TTBAR_THEORY) on the processes that carry TTBAR_THEORY_WEIGHT, whose samples are
    `samples`: the shifted weight over its inclusive normalisation of the ttbar samples; in the nominal region."""
    ((weight, _),) = TTBAR_THEORY_WEIGHT.items()
    return _theory_variations(TTBAR_THEORY, weight, era, tuple(samples))


def _largest(a: str, b: str, c: str) -> str:
    """max(a, b, c) of three columns: exactly one term is not zero."""
    return f"((({a} >= {b}) & ({a} >= {c})) * {a} + (({b} > {a}) & ({b} >= {c})) * {b} + (({c} > {a}) & ({c} > {b})) * {c})"


def _smallest(a: str, b: str, c: str) -> str:
    """min(a, b, c) of three columns: exactly one term is not zero."""
    return f"((({a} <= {b}) & ({a} <= {c})) * {a} + (({b} < {a}) & ({b} <= {c})) * {b} + (({c} < {a}) & ({c} < {b})) * {c})"


def tau_highpt_variations(channel: str, era: str) -> tuple[WeightVariation, ...]:
    """CMS_eff_t_highpT_<era> (TAU_HIGHPT) on the simulated processes of a tau channel: the vsJet weight with, for
    every genuine tau_h leg above 140 GeV, the envelope SF in place of the DM-binned one; every other leg keeps its
    weight (weights.tau_id_weight). In the nominal region."""
    name = nuisance(TAU_HIGHPT, "tau_highpt", era)
    result = ()
    for direction, envelope, bound in (("Up", _largest, "up"), ("Down", _smallest, "down")):
        legs = []
        for leg in TAU_LEGS[channel]:
            columns = crown.highpt_columns(TAU_VS_JET_WP, leg)
            nominal = f"id_wgt_tau_vsJet_{TAU_VS_JET_WP}_{leg}"
            genuine, high = f"(gen_match_{leg} == 5)", f"(pt_{leg} > {crown.HIGHPT_THRESHOLD!r})"
            legs.append(
                f"(({genuine} & {high}) * {envelope(columns[bound], columns['nominal'], nominal)}"
                f" + ({genuine} & ~{high}) * {nominal} + (gen_match_{leg} != 5))"
            )
        result += (WeightVariation(f"{name}{direction}", {TAU_ID_WEIGHT: " * ".join(legs)}, regions=(NOMINAL,)),)
    return result


@dataclass(frozen=True)
class CrownShift:
    """One CROWN shift (an Up/Down pair) of the v8 production and what the datacards make of it."""

    crown: str  # the CROWN shift without its direction (crown_names)
    nuisance: str | None  # the datacard nuisance; None: declared without nuisance (SH26), never filled
    kinematic: bool  # it moves the kinematics (the friends must follow it, SH19); else a weight
    kinds: tuple[str, ...] = MC  # the sample kinds with the shift
    samples: str = ALL  # ALL, RECOIL (the recoil-corrected samples) or NO_RECOIL (every other one)
    part: str | None = None  # the part token where the nuisance is a sum of shifts (v8_variation_sums)


def _embedding_names(channel: str, era: str) -> dict[str, tuple[str, str]]:
    """Decay-mode category of the embedding tau corrections -> its (ES, vsJet ID) nuisance; the vsEle working point in
    the name decorrelates et (Tight) from mt and tt (VVLoose)."""
    wp = f"vsEle{TAU_VS_ELE_WP[channel]}"
    return {token: (nuisance(f"CMS_scale_t_emb_dm{token}_{wp}", "embedding", era), nuisance(f"CMS_eff_t_emb_dm{token}_{wp}", "embedding", era)) for token in EMBEDDING_TAU_DECAY_MODES}


def crown_shifts_v8(channel: str, era: str, sample_era: str | None = None, jer_regions: bool = False) -> tuple[CrownShift, ...]:
    """Every CROWN shift of the v8 production in a channel (et, mt, tt or em) with its datacard nuisance: the table
    of v8 spec 6.1. The CROWN names carry `sample_era`, the era of the n-tuples (of a borrowed sample its source era),
    the nuisances the token of the run's era (correlations.py). `jer_regions`: the six JER regions as nuisances instead
    of the parts of their sum.

    Names: where the table of the spec gives one it is used (CMS_scale_t_dm<DM>_<era>, CMS_fake_e_*, CMS_fake_m_*, the
    13TeV names of the electron ES, the recoil names of the KIT legacy); the other families (the vsJet components, the
    e -> tau and mu -> tau ES) take the name of the CMS list, which is the CROWN name, with the token of the family
    (V32). Declared without nuisance (V7, V25, SH26): the lepton ID, isolation and trigger SFs (lnN as before), the
    QCD scale and the parton shower (nuisances of the signal and of ttbar only, as weights normalised to the inclusive
    rate: signal_theory_variations, ttbar_theory_variations) and the muon energy scale."""
    source = sample_era or era
    tau_channel = channel in TAU_CHANNELS

    def name(base: str, family: str) -> str:
        return nuisance(base, family, era)

    shifts = [CrownShift(crown.jes(jes), name(f"CMS_scale_j_{jes}", "jes_correlated"), True) for jes in crown.JES_CORRELATED]
    shifts += [CrownShift(crown.jes(jes, source), name(f"CMS_scale_j_{jes}", "jes_year"), True) for jes in crown.JES_UNCORRELATED]
    # V23: one JER nuisance per era, the sum of the six regions; the regions themselves with jer_regions
    shifts += [
        CrownShift(crown.jer(region, source), name(f"CMS_res_j_{region}", "jer"), True) if jer_regions else CrownShift(crown.jer(region, source), name("CMS_res_j", "jer"), True, part=region)
        for region in crown.JER_REGIONS
    ]
    # V9: the recoil-corrected samples carry the recoil nuisances and no unclustered energy, every other sample the reverse
    recoil_era = source in RECOIL_ERAS
    shifts.append(CrownShift(crown.unclustered_energy(source), name("CMS_scale_met_unclustered_energy", "met_unclustered"), True, samples=NO_RECOIL if recoil_era else ALL))
    if recoil_era:
        shifts += [CrownShift(crown.recoil(kind, source), name(f"CMS_htt_boson_{kind}_met", "recoil"), True, samples=RECOIL) for kind in ("scale", "res")]
    shifts += [CrownShift(crown.electron_es(kind, source), name(f"CMS_{kind}_e_13TeV", "electron_es"), True) for kind in ("scale", "res")]
    # V25: the muon energy scale is produced (simulation and embedding) but no nuisance in S1
    shifts += [CrownShift(crown.muon_es(region, source), None, True, kinds=MC_AND_EMB) for region in crown.MUON_ES_REGIONS.get(channel, ())]
    if tau_channel:
        shifts += [CrownShift(crown.tau_es(dm, source), name(f"CMS_scale_t_dm{dm}", "tau_mc"), True) for dm in crown.DECAY_MODES]
        if channel == "et":
            shifts += [
                CrownShift(crown.electron_fake_es(dm, region, source), name(f"CMS_scale_t_{crown.TAU_ID}_DM{dm}_genElectron_{region}", "tau_e_es"), True)
                for dm in crown.DECAY_MODES for region in crown.ETA_REGIONS
            ]
        if channel == "mt":
            shifts += [CrownShift(crown.muon_fake_es(dm, source), name(f"CMS_scale_t_{crown.TAU_ID}_DM{dm}_genMuon", "tau_mc"), True) for dm in crown.DECAY_MODES]
        vsjet = f"CMS_eff_t_{crown.TAU_ID}_VSjet_dm"
        shifts += [CrownShift(crown.vsjet_stat(n, dm, source), name(f"{vsjet}_stat{n}_DM{dm}", "tau_mc"), False) for n in crown.VSJET_STAT for dm in crown.DECAY_MODES]
        shifts.append(CrownShift(crown.vsjet_syst_alleras(), name(f"{vsjet}_syst_alleras", "tau_vsjet_alleras"), False))
        shifts.append(CrownShift(crown.vsjet_syst(source), name(f"{vsjet}_syst", "tau_mc"), False))
        shifts += [CrownShift(crown.vsjet_syst_tes(dm, source), name(f"{vsjet}_syst_TES_dm{dm}", "tau_mc"), False) for dm in crown.DECAY_MODES]
        # S10: they act on e and mu faking a tau_h only; the lnN eff_t_vsLep of genuine taus stays
        shifts += [CrownShift(crown.vsele(dm, region, source), name(f"CMS_fake_e_dm{dm}_{region}", "tau_vse"), False) for dm in crown.DECAY_MODES for region in crown.ETA_REGIONS]
        shifts += [CrownShift(crown.vsmu(wheel, source), name(f"CMS_fake_m_WH{wheel}", "tau_vsmu"), False) for wheel in crown.WHEELS]
    shifts.append(CrownShift(crown.pileup(source), name("CMS_pileup", "pileup"), False))
    shifts += [CrownShift(shift, None, False) for shift in crown.QCD_SCALES + crown.PARTON_SHOWER]
    shifts += [CrownShift(shift, None, False) for shift in crown.lepton_sf(channel, source)]  # V7: lnN as before
    if channel == "tt":  # Q18, C39: four shapes per era instead of the lnN per decay mode
        shifts += [CrownShift(crown.ditau_trigger(dm, source), name(f"CMS_eff_trigger_tt_dm{dm}", "trigger"), False) for dm in crown.DECAY_MODES]
    if tau_channel:
        for token, (energy_scale, tau_id) in _embedding_names(channel, era).items():
            decay_modes = EMBEDDING_TAU_DECAY_MODES[token]
            for dm in decay_modes:
                # V20: every decay mode is a part, of its own nuisance and of the one shared with the simulation
                shifts.append(CrownShift(crown.embedding_tau_es(dm, source), energy_scale, True, kinds=EMB, part=f"dm{dm}"))
                shifts.append(CrownShift(crown.embedding_tau_id(dm, source), tau_id, False, kinds=EMB, part=f"dm{dm}" if len(decay_modes) > 1 else None))
        if channel == "et":  # V22, V34: the same names in every era
            shifts += [CrownShift(crown.embedding_electron_es(region, source), name(f"CMS_scale_e_{region}_emb", "embedding"), True, kinds=EMB) for region in crown.ETA_REGIONS]
    return tuple(shifts)


def _shift_groups(shift: CrownShift, groups: Mapping[str, tuple[str, str]], sample_era: str) -> tuple[str, ...]:
    """The sample groups of the channel that carry the shift in the n-tuples of `sample_era`."""
    chosen = [group for group, (kind, era) in groups.items() if kind in shift.kinds and era == sample_era]
    if shift.samples == RECOIL:
        return tuple(group for group in chosen if group in RECOIL_GROUPS)
    return tuple(group for group in chosen if shift.samples == ALL or group not in RECOIL_GROUPS)


def v8_variations(channel: str, era: str, groups: Mapping[str, tuple[str, str]], nuisances: bool = True, required: tuple[str, ...] = (), jer_regions: bool = False) -> tuple[ColumnVariation, ...]:
    """The column variations of the v8 shifts of a channel (crown_shifts_v8). `groups`: sample group -> (kind, era
    of its n-tuples) of the simulated and embedded samples of the channel; a group borrowed from another era reads the
    shifts under the names of that era into the nuisances of the run's era.

    With `nuisances` a shift with a nuisance is filled in the nominal region only (V31: the anti-isolated region books
    no CROWN shift, so the jet fakes carry the fake-factor uncertainties alone), under the name of its nuisance or of its
    part; a kinematic one requires `required`, the friend columns that must follow it (SH19). Every other shift, and
    without `nuisances` all of them (the S0 cards of a production with shifts), is declared under its CROWN name without
    regions (SH26): the skim accepts its branches, nothing is filled."""
    result = ()
    kinds = {kind for kind, _ in groups.values()}
    for sample_era in dict.fromkeys(sample_era for _, sample_era in groups.values()):
        for shift in crown_shifts_v8(channel, era, sample_era, jer_regions):
            chosen = _shift_groups(shift, groups, sample_era)
            if not chosen:
                continue
            everyone = tuple(group for group, (kind, _) in groups.items() if kind in shift.kinds)
            applies_to = tuple(kind for kind in shift.kinds if kind in kinds)
            filled = nuisances and shift.nuisance is not None
            for direction in DIRECTIONS:
                if not filled:
                    variation = f"{shift.crown}{direction}"
                elif shift.part is not None:
                    variation = part_of(f"{shift.nuisance}{direction}", shift.part)
                else:
                    variation = f"{shift.nuisance}{direction}"
                result += (ColumnVariation(
                    variation, crown.suffix(shift.crown, direction), applies_to=applies_to, regions=(NOMINAL,) if filled else (),
                    groups=None if chosen == everyone else chosen, required=required if filled and shift.kinematic else (),
                ),)
    return result


def v8_variation_sums(channel: str, era: str, embedded: bool, jer_regions: bool = False) -> tuple[VariationSum, ...]:
    """The nuisances of the v8 shifts that are sums of CROWN shifts (the parts of crown_shifts_v8):
    - CMS_res_j_<era>: the six JER regions (V23; none with jer_regions);
    - in an embedded tau channel the embedding tau ES per decay-mode category with EMBEDDING_OWN of its shift, and
      CMS_scale_t_dm<DM>_<era>, the nuisance of the simulation, with EMBEDDING_SHARED of the embedding shift of the decay
      mode on the embedded sample (V20: 50 % correlated, 0.5^2 + 0.866^2 = 1);
    - the embedding vsJet ID of DM10 and DM11, one measured category."""
    result = () if jer_regions else (VariationSum(nuisance("CMS_res_j", "jer", era), crown.JER_REGIONS),)
    if not (embedded and channel in TAU_CHANNELS):
        return result
    for token, (energy_scale, tau_id) in _embedding_names(channel, era).items():
        decay_modes = EMBEDDING_TAU_DECAY_MODES[token]
        result += (VariationSum(energy_scale, tuple(f"dm{dm}" for dm in decay_modes), scale=EMBEDDING_OWN),)
        result += tuple(VariationSum(nuisance(f"CMS_scale_t_dm{dm}", "tau_mc", era), (f"dm{dm}",), scale=EMBEDDING_SHARED, source=energy_scale) for dm in decay_modes)
        if len(decay_modes) > 1:
            result += (VariationSum(tau_id, tuple(f"dm{dm}" for dm in decay_modes)),)
    return result


def lumi_lnn(processes: tuple[str, ...], era: str) -> tuple[LnN, ...]:
    """The Cholesky lumi nuisances of the era's year (spec 6.6)."""
    year = YEAR[era]
    return tuple(LnN(name, processes, values[year]) for name, values in LUMI.items() if year in values)


@dataclass(frozen=True)
class Study:
    """The run's view of the switch study_uncertainties: its embedded tau channels, its fake-factor variant (None
    without fake factors) and which CROWN shifts are nuisances of the run, since a shift replaces its stand-in (SH17):
    `crown_shifts` the embedding tau ID and ES (the shifts of every production with shifts), `mc_shifts` the tau ES,
    vsJet components and di-tau trigger SF of the simulation (the v8 shifts)."""

    embedded: tuple[str, ...]
    ff_variant: str | None
    crown_shifts: bool
    mc_shifts: bool = False


def ff_variant(wjets: bool) -> str:
    return "QCD_Wjets_ttbar" if wjets else "QCD_ttbar"


def _decay_modes(column: str, decay_modes: tuple[int, ...]) -> str:
    return " | ".join(f"({column} == {dm})" for dm in decay_modes)


def embedding_id_variations(channel: str, era: str) -> tuple[ColumnVariation, ...]:
    """The embedding vsJet SF interval of each decay-mode category (study_uncertainties.EMBEDDING_TAU_ID) as exact
    weight factors on the SF column of the embedded sample, under the names of the CROWN shifts (embedding_variations);
    the stand-in of a nominal-only production. One tau per event (et, mt): a category of two decay modes is one factor.
    The fake-factor estimate subtracts the embedded sample in the AR under it too."""
    column = f"id_wgt_tau_vsJet_{TAU_VS_JET_WP}_2"
    result = ()
    for token, decay_modes in EMBEDDING_TAU_DECAY_MODES.items():
        name = nuisance(f"CMS_eff_t_emb_dm{token}_vsEle{TAU_VS_ELE_WP[channel]}", "embedding", era)
        down, up = study_sizes.EMBEDDING_TAU_ID[era][TAU_VS_ELE_WP[channel]][token]
        for direction, ratio in (("Up", up), ("Down", down)):
            factor = f"(1 + ({_decay_modes('tau_decaymode_2', decay_modes)}) * {round(ratio - 1, 4)!r})"
            result += (ColumnVariation(f"{name}{direction}", derived={column: f"{column} * {factor}"}, applies_to=("embedding",)),)
    return result


def ff_dm_nonclosure_variations(channel: str, era: str, variant: str) -> tuple[ColumnVariation, ...]:
    """The residual decay-mode non-closure of the fake factors after all corrections (study_uncertainties
    FF_DM_NONCLOSURE): Up multiplies the fake factor of each leg by 1 + shift(DM of the leg), Down by 1 - shift; per
    channel and year (one 2016 measurement). Channels whose decay-mode closure holds (et) have none."""
    table = study_sizes.FF_DM_NONCLOSURE[variant][YEAR[era]].get(channel)
    if table is None:
        return ()
    name = nuisance(f"CMS_ff_dm_nonclosure_{channel}", "fake_factors", era)
    result = ()
    for direction, sign in (("Up", 1), ("Down", -1)):
        derived = {}
        for leg in TAU_LEGS[channel]:
            column = FF_COLUMNS["lt"] if channel in LT_CHANNELS else FF_COLUMNS[f"tt_{leg}"]
            dm = f"tau_decaymode_{leg}"
            bins = {0: f"({dm} == 0)", 1: f"(({dm} == 1) | ({dm} == 2))", 10: f"({dm} == 10)", 11: f"({dm} == 11)"}
            shift = " + ".join(f"{bins[d]} * {sign * value!r}" for d, value in table[str(leg)].items())
            derived[column] = f"{column} * (1 + {shift})"
        result += (ColumnVariation(f"{name}{direction}", derived=derived, applies_to=("data", "mc", "embedding")),)
    return result


def study_lnn(processes: tuple[Process, ...], jet_fakes_output: str, era: str, study: Study, cms_names: bool = False) -> list[LnN]:
    """The lnN of the switch study_uncertainties (study_uncertainties.py, systematics_v7b_summary.md).
    Stand-ins, each left out where the CROWN shifts of the run take its place (Study):
    - embedding tau ES per decay-mode category (et, mt);
    - MC: POG tau ES per decay mode, the POG vsJet SF as syst_alleras (K), era part (E) and syst_TES per decay mode
      (E; they replace the flat eff_t_vsJet), in tt the POG di-tau trigger SF per decay mode (E), each per process class.
    lnN that stay with the shifts (v8 spec 6.2):
    - K4: muon_Z ID + isolation stat per era (mt, em), EGM electron reconstruction (K; et, em) on MC;
    - DY and W in et and mt of the eras without recoil correction: +-3 % mT acceptance (MET residual);
    - 2016preVFP: the HIPM lepton efficiency per light lepton on the MC except DY (whose table absorbs it);
    - jet fakes of the 2016 halves: the difference of the combined 2016 payload to the payload of the half.
    `cms_names`: the electron reconstruction under its name of the CMS list (SH27), CMS_eff_e_reco_13TeV."""
    mc = tuple(p.name for p in processes if p.role in ("signal", "background") and p.group != EMBEDDED)
    names = {p.name for p in processes}

    def group(*groups: str) -> tuple[str, ...]:
        return tuple(p.name for p in processes if p.group in groups and p.role != "auxiliary")

    entries = []
    if not study.crown_shifts:
        for channel in (c for c in study.embedded if c in LT_CHANNELS):
            for token, value in study_sizes.EMBEDDING_TAU_ES[era][channel].items():
                name = nuisance(f"CMS_scale_t_emb_dm{token}_vsEle{TAU_VS_ELE_WP[channel]}", "embedding", era)
                entries.append(LnN(name, (EMBEDDED,), value, channels=(channel,)))
    for channel, classes in ({} if study.mc_shifts else study_sizes.MC_TAU_CLASSES).items():
        for klass, members in classes.items():
            table = study_sizes.MC_TAU[era][channel].get(klass)
            members = tuple(m for m in members if m in names and m in mc)
            if table is None or not members:
                continue
            channels = (channel,)
            for dm in (0, 1, 10, 11):
                entries.append(LnN(nuisance(f"CMS_scale_t_dm{dm}", "tau_mc", era), members, table[f"tes_dm{dm}"], channels=channels))
                entries.append(LnN(nuisance(f"CMS_eff_t_vsJet_syst_TES_dm{dm}", "tau_mc", era), members, table[f"syst_TES_dm{dm}"], channels=channels))
                if f"trigger_dm{dm}" in table:
                    entries.append(LnN(nuisance(f"CMS_eff_t_trigger_ditau_dm{dm}", "trigger", era), members, table[f"trigger_dm{dm}"], channels=channels))
            entries.append(LnN(nuisance("CMS_eff_t_vsJet_syst_alleras", "tau_vsjet_alleras", era), members, table["syst_alleras"], channels=channels))
            entries.append(LnN(nuisance("CMS_eff_t_vsJet", "tau_mc", era), members, table["era"], channels=channels))
    for channel, sizes in study_sizes.LEPTONS[era].items():
        if "muon_stat" in sizes:
            # OPEN (SH27): the CMS list has CMS_eff_m_id_stat_<era> and CMS_eff_m_iso_stat_<era>; this lnN is the linear
            # sum of both parts (0.04-0.07 %), whose separate sizes the study did not keep, so it keeps its one name
            entries.append(LnN(nuisance("CMS_eff_m_stat", "muon_id_stat", era), mc, sizes["muon_stat"], channels=(channel,)))
        if "electron_reco" in sizes:
            entries.append(LnN(nuisance("CMS_eff_e_reco_13TeV" if cms_names else "CMS_eff_e_reco", "electron_reco", era), mc, sizes["electron_reco"], channels=(channel,)))
    if era not in RECOIL_ERAS:
        entries.append(LnN(nuisance("CMS_htt_met_mt_acceptance", "met_residual", era), group("DY", "W"), study_sizes.MET_RESIDUAL, channels=LT_CHANNELS))
    if era in study_sizes.HIPM:
        without_dy = tuple(name for name in mc if name not in group("DY"))
        hipm = study_sizes.HIPM[era]
        entries.append(LnN(nuisance("CMS_htt_hipm_eff_m", "hipm", era), without_dy, hipm["m"], channels=("mt", "em")))
        entries.append(LnN(nuisance("CMS_htt_hipm_eff_e", "hipm", era), without_dy, hipm["e"], channels=("et", "em")))
    halves = study_sizes.FF_HALVES.get(study.ff_variant, {}).get(era, {}) if jet_fakes_output == "jetFakes" else {}
    for channel, value in halves.items():
        entries.append(LnN(nuisance(f"CMS_ff_2016half_{channel}", "ff_halves", era), (jet_fakes_output,), value, channels=(channel,)))
    return [e for e in entries if _relevant(e.value)]


def _relevant(value: float | tuple[float, float]) -> bool:
    """An lnN that moves the yield by more than 0.01 % in one direction (the others are left out of the card)."""
    return any(abs(v - 1) > 0.0001 + 1e-9 for v in (value if isinstance(value, tuple) else (value,)))


def dy_table_lnn(processes: tuple[Process, ...], era: str) -> LnN:
    """The DY-table nuisance of the era (constants.DY_TABLE_LNN) on every DY process the table corrects."""
    if era not in DY_TABLE_LNN:
        raise AnalysisError(f"dy_correction njets_btag: no DY-table uncertainty for {era} (constants.DY_TABLE_LNN has {', '.join(DY_TABLE_LNN)})")
    dy = tuple(p.name for p in processes if p.group == "DY" and p.role != "auxiliary")
    return LnN(nuisance("CMS_htt_dy_table", "dy_table", era), dy, DY_TABLE_LNN[era])


def lnn(processes: tuple[Process, ...], jet_fakes_output: str, era: str, borrowed: tuple[str, ...] = (), study: Study | None = None, dy_correction: str = "none",
        mc_shifts: bool = False) -> tuple[LnN, ...]:
    """Normalisation uncertainties; `borrowed` are the processes made of samples borrowed from another era; `study`
    adds the lnN of the switch study_uncertainties (study_lnn), whose POG vsJet components replace the flat
    eff_t_vsJet; with the DY table (dy_correction njets_btag) its uncertainty (dy_table_lnn). `mc_shifts`: the v8
    shifts of the simulation are nuisances of the run, so their vsJet shapes replace the flat eff_t_vsJet too, and the
    lnN take the names of the CMS list (SH27). The lepton ID and trigger lnN stay with the shifts (V7), as does
    eff_t_vsLep of genuine taus (S10: the vsEle and vsMu shifts act on lepton fakes only)."""
    mc = tuple(p.name for p in processes if p.role in ("signal", "background") and p.group != EMBEDDED)
    emb = tuple(p.name for p in processes if p.role == "background" and p.group == EMBEDDED)

    def group(*groups: str) -> tuple[str, ...]:
        return tuple(p.name for p in processes if p.group in groups and p.role != "auxiliary")

    def trigger(name: str) -> str:
        return nuisance(name, "trigger", era)

    def tau(name: str) -> str:
        return nuisance(name, "tau_mc", era)

    fakes_family = "fake_factors" if jet_fakes_output == "jetFakes" else "qcd"
    entries = [
        *lumi_lnn(mc, era),
        # lepton ID/iso: 2 % on MC; on the embedded sample half of it shared and its own 0.866 part (E27)
        LnN("eff_e", mc, LEPTON_ID, channels=("et", "em")),
        LnN("eff_m", mc, LEPTON_ID, channels=("mt", "em")),
        LnN("eff_e", emb, round(1 + EMBEDDING_SHARED * (LEPTON_ID - 1), 3), channels=("et",)),
        LnN("eff_m", emb, round(1 + EMBEDDING_SHARED * (LEPTON_ID - 1), 3), channels=("mt",)),
        LnN("eff_e_emb", emb, round(1 + EMBEDDING_OWN * (LEPTON_ID - 1), 3), channels=("et",)),
        LnN("eff_m_emb", emb, round(1 + EMBEDDING_OWN * (LEPTON_ID - 1), 3), channels=("mt",)),
        # single-lepton triggers, per era, MC and embedding separately (KIT SFs, except the POG muon_Z SF of the MC muon
        # trigger from smrun2_binned_v7b on, E5); em triggers on the muon
        LnN(trigger("CMS_eff_m_trigger"), mc, TRIGGER_LNN, channels=("mt", "em")),
        LnN(trigger("CMS_eff_m_trigger_emb"), emb, TRIGGER_LNN, channels=("mt",)),
        LnN(trigger("CMS_eff_e_trigger"), mc, TRIGGER_LNN, channels=("et",)),
        LnN(trigger("CMS_eff_e_trigger_emb"), emb, TRIGGER_LNN, channels=("et",)),
        # VSe and VSmu together (K6: the 2016 eras, whose VSe and VSmu payloads are 2018 copies, share the 2018 nuisance)
        LnN(nuisance("eff_t_vsLep_$CHANNEL", "tau_vslep", era), mc, 1.01, channels=LT_CHANNELS),
        LnN(nuisance("eff_t_vsLep_$CHANNEL", "tau_vslep", era), mc, 1.0201, channels=("tt",)),
        *(() if study or mc_shifts else (LnN(tau("eff_t_vsJet_$CHANNEL"), mc, 1.02, channels=LT_CHANNELS), LnN(tau("eff_t_vsJet_$CHANNEL"), mc, 1.0404, channels=("tt",)))),
        LnN("htt_zjXsec", group("DY"), 1.02),
        LnN("htt_tjXsec", group("TT"), 1.06),
        LnN("htt_stXsec", group("ST"), 1.05),
        LnN("htt_vvXsec", group("VV"), 1.05),
        LnN("htt_ttvXsec", group("TTV"), 1.10),
        LnN("htt_ewkXsec", group("EWK"), 1.05),
        LnN("htt_wjXsec", group("W"), 1.04),
        LnN("QCDscale_ggH", group("ggH"), 1.039), LnN("pdf_ggH", group("ggH"), 1.019), LnN("alphaS_ggH", group("ggH"), 1.026),
        LnN("QCDscale_qqH", group("qqH"), (0.997, 1.004)), LnN("pdf_qqH", group("qqH"), 1.021), LnN("alphaS_qqH", group("qqH"), 1.005),
        LnN("QCDscale_ttH", group("ttH"), (0.908, 1.058)), LnN("pdf_ttH", group("ttH"), 1.030), LnN("alphaS_ttH", group("ttH"), 1.020),
        LnN("QCDscale_VH", group("VH", VH_TAUTAU), (0.970, 1.038)), LnN("pdf_VH", group("VH", VH_TAUTAU), 1.017), LnN("alphaS_VH", group("VH", VH_TAUTAU), 1.009),
        # HH (E26, LHCHWGHHRecommendations 2026-09-15): scale + THnum + m_t +7.25/-21.12 %, PDF + alpha_s +-2.47 %
        LnN("QCDscale_HH", (SIGNAL,), (0.7888, 1.0725)), LnN("PDF_alphas_HH", (SIGNAL,), 1.0247),
        LnN("BR_h_bb", (SIGNAL,), (0.9873, 1.0125)), LnN("BR_h_tautau", (SIGNAL,), 1.0165),
        LnN(nuisance(f"{jet_fakes_output}Norm_$CHANNEL", fakes_family, era), (jet_fakes_output,), 1.108, channels=LT_CHANNELS),
        LnN(nuisance(f"{jet_fakes_output}Norm_$CHANNEL", fakes_family, era), (jet_fakes_output,), 1.216, channels=("tt",)),
        *(LnN(f"CMS_htt_borrowed_{name}_{era}", (name,), BORROWED_LNN) for name in borrowed),
        *(study_lnn(processes, jet_fakes_output, era, study, cms_names=mc_shifts) if study else ()),
        *((dy_table_lnn(processes, era),) if dy_correction == "njets_btag" else ()),
    ]
    return tuple(e for e in entries if e.processes)
