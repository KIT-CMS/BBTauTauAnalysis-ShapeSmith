"""Systematic uncertainties: b-tag weight variations, the CROWN shifts of the embedded sample and of the fake-factor
friend (column variations), normalisation uncertainties (lnN)."""
from __future__ import annotations

from collections.abc import Iterable

from shapesmith.histogram import part_of
from shapesmith.model import ColumnVariation, LnN, Process, VariationSum, WeightVariation

from bbtautau_shapesmith.constants import BTAG_COMPONENTS, EMBEDDING_TAU_DECAY_MODES, EMBEDDING_VS_JET_PT_BINS, ERA, FF_SHIFTS_LT, FF_SHIFTS_TT, LT_CHANNELS, TAU_VS_ELE_WP
from bbtautau_shapesmith.processes import EMBEDDED, SIGNAL


def btag_variations(components: Iterable[str] = BTAG_COMPONENTS) -> tuple[WeightVariation, ...]:
    """Up/Down shape variations of the UParT b-tag weight, one pair per component."""
    return tuple(
        WeightVariation(f"CMS_btag_{component}_{ERA}{shift}", {"btag": f"btag_weight_upart_{shift.lower()}_{component}"})
        for component in components
        for shift in ("Up", "Down")
    )


def _shifts(name: str, suffix: str, applies_to: tuple[str, ...]) -> tuple[ColumnVariation, ...]:
    """The Up/Down pair of one CROWN shift: datacard name `name`Up/Down, branches `c + suffix`Up/Down."""
    return tuple(ColumnVariation(f"{name}{shift}", f"{suffix}{shift}", applies_to=applies_to) for shift in ("Up", "Down"))


def _embedding_nuisances(channel: str) -> list[tuple[str, dict[int, str]]]:
    """(datacard name, decay mode -> CROWN shift) of the embedding tau corrections, one per measured category. The
    vsEle working point in the name decorrelates et (Tight) from mt and tt (VVLoose)."""
    wp = f"vsEle{TAU_VS_ELE_WP[channel]}"
    result = []
    for token, decay_modes in EMBEDDING_TAU_DECAY_MODES.items():
        shifts = {dm: f"CMS_scale_t_emb_DeepTau2018v2p5_DM{dm}_{ERA}" for dm in decay_modes}
        result.append((f"CMS_scale_t_emb_dm{token}_{wp}_{ERA}", shifts))
    for token, decay_modes in EMBEDDING_TAU_DECAY_MODES.items():
        for pt in EMBEDDING_VS_JET_PT_BINS[channel]:
            shifts = {dm: f"CMS_eff_t_emb_DeepTau2018v2p5_VSjet_DM{dm}_pt{pt}_{ERA}" for dm in decay_modes}
            result.append((f"CMS_eff_t_emb_dm{token}_pt{pt}_{wp}_{ERA}", shifts))
    return result


def embedding_variations(channel: str) -> tuple[ColumnVariation, ...]:
    """Tau ES and vsJet ID shifts of the embedded sample. A nuisance of one decay mode reads its CROWN shift; one of
    two decay modes reads each as a part of the sum of embedding_variation_sums."""
    result = ()
    for name, shifts in _embedding_nuisances(channel):
        if len(shifts) == 1:
            (shift,) = shifts.values()
            result += _shifts(name, f"__{shift}", ("embedding",))
            continue
        result += tuple(
            ColumnVariation(part_of(f"{name}{d}", f"dm{dm}"), f"__{shift}{d}", applies_to=("embedding",))
            for dm, shift in shifts.items()
            for d in ("Up", "Down")
        )
    return result


def embedding_variation_sums(channel: str) -> tuple[VariationSum, ...]:
    """The nuisances of two decay modes (DM10 and DM11), each the sum of its per-decay-mode CROWN shifts: exact in et
    and mt (one tau per event), to first order in tt."""
    return tuple(VariationSum(name, tuple(f"dm{dm}" for dm in shifts)) for name, shifts in _embedding_nuisances(channel) if len(shifts) > 1)


def ff_variations(channel: str) -> tuple[ColumnVariation, ...]:
    """The fake-factor friend shifts, per channel; in tt a key shifts only the fake factor of its leg."""
    keys = FF_SHIFTS_TT if channel == "tt" else FF_SHIFTS_LT
    return tuple(variation for key in keys for variation in _shifts(f"CMS_ff_{key}_{channel}_{ERA}", f"__{key}", ("data", "mc", "embedding")))


def lnn(processes: tuple[Process, ...], jet_fakes_output: str) -> tuple[LnN, ...]:
    mc = tuple(p.name for p in processes if p.role in ("signal", "background") and p.group != EMBEDDED)

    def group(*groups: str) -> tuple[str, ...]:
        return tuple(p.name for p in processes if p.group in groups and p.role != "auxiliary")

    entries = [
        LnN("lumi_13TeV_$ERA", mc, 1.025),
        LnN("eff_e", mc, 1.02, channels=("et",)),
        LnN("eff_m", mc, 1.02, channels=("mt",)),
        LnN("eff_t_vsLep_$CHANNEL_$ERA", mc, 1.01, channels=LT_CHANNELS),
        LnN("eff_t_vsLep_$CHANNEL_$ERA", mc, 1.0201, channels=("tt",)),
        LnN("eff_t_vsJet_$CHANNEL_$ERA", mc, 1.02, channels=LT_CHANNELS),
        LnN("eff_t_vsJet_$CHANNEL_$ERA", mc, 1.0404, channels=("tt",)),
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
        LnN("QCDscale_VH", group("VH"), (0.970, 1.038)), LnN("pdf_VH", group("VH"), 1.017), LnN("alphaS_VH", group("VH"), 1.009),
        LnN("QCDscale_HH", (SIGNAL,), (0.95, 1.022)), LnN("PDF_alphas_HH", (SIGNAL,), 1.03), LnN("mtop_HH", (SIGNAL,), 1.026),
        LnN("BR_h_bb", (SIGNAL,), (0.9873, 1.0125)), LnN("BR_h_tautau", (SIGNAL,), 1.0165),
        LnN(f"{jet_fakes_output}Norm_$CHANNEL_$ERA", (jet_fakes_output,), 1.108, channels=LT_CHANNELS),
        LnN(f"{jet_fakes_output}Norm_$CHANNEL_$ERA", (jet_fakes_output,), 1.216, channels=("tt",)),
    ]
    return tuple(e for e in entries if e.processes)
