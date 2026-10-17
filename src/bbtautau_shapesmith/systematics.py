"""Systematic uncertainties per era: b-tag and prefiring weight variations, the CROWN shifts of the embedded sample and
of the fake-factor friend (column variations), normalisation uncertainties (lnN). The era tokens follow the
correlation scheme of correlations.py (v7 spec 6.7); in v7 (nominal only, E34) the datacards carry the lnN and the
variations of the nominal trees (b tag, fake factors, prefiring)."""
from __future__ import annotations

from shapesmith.histogram import part_of
from shapesmith.model import ColumnVariation, LnN, Process, VariationSum, WeightVariation

from bbtautau_shapesmith.constants import (
    BTAG_COMPONENTS, EMBEDDING_TAU_DECAY_MODES, FF_SHIFTS_LT, FF_SHIFTS_TT, LT_CHANNELS, PREFIRING_NUISANCE, PREFIRING_VARIATIONS, TAU_VS_ELE_WP, YEAR,
)
from bbtautau_shapesmith.correlations import nuisance
from bbtautau_shapesmith.processes import EMBEDDED, SIGNAL, VH_TAUTAU

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
TRIGGER_LNN = 1.02  # the KIT single-lepton trigger SFs have no variations: +-2 % (TauAnalysis convention, spec C6)
BORROWED_LNN = 1.05  # a process of samples borrowed from another era (3.3): the detector differences


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


def ff_variations(channel: str, era: str) -> tuple[ColumnVariation, ...]:
    """The fake-factor friend shifts, per channel and year (2016 is measured on both halves); in tt a key shifts only
    the fake factor of its leg."""
    keys = FF_SHIFTS_TT if channel == "tt" else FF_SHIFTS_LT
    return tuple(variation for key in keys for variation in _shifts(nuisance(f"CMS_ff_{key}_{channel}", "fake_factors", era), f"__{key}", ("data", "mc", "embedding")))


def lumi_lnn(processes: tuple[str, ...], era: str) -> tuple[LnN, ...]:
    """The Cholesky lumi nuisances of the era's year (spec 6.6)."""
    year = YEAR[era]
    return tuple(LnN(name, processes, values[year]) for name, values in LUMI.items() if year in values)


def lnn(processes: tuple[Process, ...], jet_fakes_output: str, era: str, borrowed: tuple[str, ...] = ()) -> tuple[LnN, ...]:
    """Normalisation uncertainties; `borrowed` are the processes made of samples borrowed from another era."""
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
        # single-lepton triggers (KIT SFs), per era, MC and embedding separately; em triggers on the muon
        LnN(trigger("CMS_eff_m_trigger"), mc, TRIGGER_LNN, channels=("mt", "em")),
        LnN(trigger("CMS_eff_m_trigger_emb"), emb, TRIGGER_LNN, channels=("mt",)),
        LnN(trigger("CMS_eff_e_trigger"), mc, TRIGGER_LNN, channels=("et",)),
        LnN(trigger("CMS_eff_e_trigger_emb"), emb, TRIGGER_LNN, channels=("et",)),
        LnN(tau("eff_t_vsLep_$CHANNEL"), mc, 1.01, channels=LT_CHANNELS),
        LnN(tau("eff_t_vsLep_$CHANNEL"), mc, 1.0201, channels=("tt",)),
        LnN(tau("eff_t_vsJet_$CHANNEL"), mc, 1.02, channels=LT_CHANNELS),
        LnN(tau("eff_t_vsJet_$CHANNEL"), mc, 1.0404, channels=("tt",)),
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
    ]
    return tuple(e for e in entries if e.processes)
