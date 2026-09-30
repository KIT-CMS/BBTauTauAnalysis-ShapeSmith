"""Systematic uncertainties: b-tag weight variations (shapes), normalisation uncertainties (lnN)."""
from __future__ import annotations

from collections.abc import Iterable

from shapesmith.model import LnN, Process, WeightVariation

from bbtautau_shapesmith.constants import BTAG_COMPONENTS, ERA
from bbtautau_shapesmith.processes import EMBEDDED, SIGNAL

LT = ("et", "mt")
TT = ("tt",)


def btag_variations(components: Iterable[str] = BTAG_COMPONENTS) -> tuple[WeightVariation, ...]:
    """Up/Down shape variations of the UParT b-tag weight, one pair per component."""
    return tuple(
        WeightVariation(f"CMS_btag_{component}_{ERA}{shift}", {"btag": f"btag_weight_upart_{shift.lower()}_{component}"})
        for component in components
        for shift in ("Up", "Down")
    )


def lnn(processes: tuple[Process, ...], jet_fakes_output: str) -> tuple[LnN, ...]:
    mc = tuple(p.name for p in processes if p.role in ("signal", "background") and p.group != EMBEDDED)

    def group(*groups: str) -> tuple[str, ...]:
        return tuple(p.name for p in processes if p.group in groups and p.role != "auxiliary")

    entries = [
        LnN("lumi_13TeV_$ERA", mc, 1.025),
        LnN("eff_e", mc, 1.02, channels=("et",)),
        LnN("eff_m", mc, 1.02, channels=("mt",)),
        LnN("eff_t_vsLep_$CHANNEL_$ERA", mc, 1.01, channels=LT),
        LnN("eff_t_vsLep_$CHANNEL_$ERA", mc, 1.0201, channels=TT),
        LnN("eff_t_vsJet_$CHANNEL_$ERA", mc, 1.02, channels=LT),
        LnN("eff_t_vsJet_$CHANNEL_$ERA", mc, 1.0404, channels=TT),
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
        LnN(f"{jet_fakes_output}Norm_$CHANNEL_$ERA", (jet_fakes_output,), 1.108, channels=LT),
        LnN(f"{jet_fakes_output}Norm_$CHANNEL_$ERA", (jet_fakes_output,), 1.216, channels=TT),
    ]
    return tuple(e for e in entries if e.processes)
