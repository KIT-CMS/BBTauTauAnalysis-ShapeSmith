"""Systematic uncertainties per era: b-tag and prefiring weight variations, the CROWN shifts of the embedded sample and
of the fake-factor friend (column variations), normalisation uncertainties (lnN). The era tokens follow the
correlation scheme of correlations.py (v7 spec 6.7); in v7 (nominal only, E34) the datacards carry the lnN and the
variations of the nominal trees (b tag, fake factors, prefiring). The switch study_uncertainties adds the uncertainties
sized in the v7b studies (study_uncertainties.py): the embedding tau ID as exact weight factors and its ES as lnN, the
POG MC tau ES, vsJet components and di-tau trigger SF as lnN, the residual decay-mode non-closure of the fake factors as
a shape, the MET residual, the HIPM lepton efficiency, the 2016 halves of the fake factors and K4 as lnN."""
from __future__ import annotations

from dataclasses import dataclass

from shapesmith.histogram import part_of
from shapesmith.model import AnalysisError, ColumnVariation, LnN, Process, VariationSum, WeightVariation

from bbtautau_shapesmith import study_uncertainties as study_sizes
from bbtautau_shapesmith.constants import (
    BTAG_COMPONENTS, DY_TABLE_LNN, EMBEDDING_TAU_DECAY_MODES, FF_COLUMNS, FF_SHIFTS_LT, FF_SHIFTS_TT, FF_SHIFTS_W, LT_CHANNELS, PREFIRING_NUISANCE, PREFIRING_VARIATIONS,
    RECOIL_ERAS, TAU_LEGS, TAU_VS_ELE_WP, TAU_VS_JET_WP, YEAR,
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
TRIGGER_LNN = 1.02  # the single-lepton trigger SFs of the nominal production have no variations: +-2 % (TauAnalysis convention, spec C6)
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


def ff_variations(channel: str, era: str, wjets: bool = False) -> tuple[ColumnVariation, ...]:
    """The fake-factor friend shifts, per channel and year (2016 is measured on both halves); in tt a key shifts only
    the fake factor of its leg. With W+jets fake factors (ff_processes) et and mt add the W+jets keys."""
    keys = FF_SHIFTS_TT if channel == "tt" else FF_SHIFTS_LT + (FF_SHIFTS_W if wjets else ())
    return tuple(variation for key in keys for variation in _shifts(nuisance(f"CMS_ff_{key}_{channel}", "fake_factors", era), f"__{key}", ("data", "mc", "embedding")))


def lumi_lnn(processes: tuple[str, ...], era: str) -> tuple[LnN, ...]:
    """The Cholesky lumi nuisances of the era's year (spec 6.6)."""
    year = YEAR[era]
    return tuple(LnN(name, processes, values[year]) for name, values in LUMI.items() if year in values)


@dataclass(frozen=True)
class Study:
    """The run's view of the switch study_uncertainties: its embedded tau channels, its fake-factor variant (None
    without fake factors) and whether the production has CROWN shifts (then the shifts replace the stand-ins of the
    embedding tau ID/ES and the MC tau ES)."""

    embedded: tuple[str, ...]
    ff_variant: str | None
    crown_shifts: bool


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


def study_lnn(processes: tuple[Process, ...], jet_fakes_output: str, era: str, study: Study) -> list[LnN]:
    """The lnN of the switch study_uncertainties (study_uncertainties.py, systematics_v7b_summary.md):
    - embedding tau ES per decay-mode category (et, mt; the stand-in of the CROWN shifts);
    - MC: POG tau ES per decay mode (stand-in of the shifts), the POG vsJet SF as syst_alleras (K), era part (E) and
      syst_TES per decay mode (E; replaces the flat eff_t_vsJet), in tt the POG di-tau trigger SF per decay mode (E),
      each per process class;
    - K4: muon_Z ID + isolation stat per era (mt, em), EGM electron reconstruction (K; et, em) on MC;
    - DY and W in et and mt of the eras without recoil correction: +-3 % mT acceptance (MET residual);
    - 2016preVFP: the HIPM lepton efficiency per light lepton on the MC except DY (whose table absorbs it);
    - jet fakes of the 2016 halves: the difference of the combined 2016 payload to the payload of the half."""
    mc = tuple(p.name for p in processes if p.role in ("signal", "background") and p.group != EMBEDDED)
    names = {p.name for p in processes}
    year = YEAR[era]

    def group(*groups: str) -> tuple[str, ...]:
        return tuple(p.name for p in processes if p.group in groups and p.role != "auxiliary")

    entries = []
    if not study.crown_shifts:
        for channel in (c for c in study.embedded if c in LT_CHANNELS):
            for token, value in study_sizes.EMBEDDING_TAU_ES[era][channel].items():
                name = nuisance(f"CMS_scale_t_emb_dm{token}_vsEle{TAU_VS_ELE_WP[channel]}", "embedding", era)
                entries.append(LnN(name, (EMBEDDED,), value, channels=(channel,)))
    for channel, classes in study_sizes.MC_TAU_CLASSES.items():
        for klass, members in classes.items():
            table = study_sizes.MC_TAU[era][channel].get(klass)
            members = tuple(m for m in members if m in names and m in mc)
            if table is None or not members:
                continue
            channels = (channel,)
            for dm in (0, 1, 10, 11):
                if not study.crown_shifts:
                    entries.append(LnN(nuisance(f"CMS_scale_t_dm{dm}", "tau_mc", era), members, table[f"tes_dm{dm}"], channels=channels))
                entries.append(LnN(nuisance(f"CMS_eff_t_vsJet_syst_TES_dm{dm}", "tau_mc", era), members, table[f"syst_TES_dm{dm}"], channels=channels))
                if f"trigger_dm{dm}" in table:
                    entries.append(LnN(nuisance(f"CMS_eff_t_trigger_ditau_dm{dm}", "trigger", era), members, table[f"trigger_dm{dm}"], channels=channels))
            entries.append(LnN(nuisance("CMS_eff_t_vsJet_syst_alleras", "tau_vsjet_alleras", era), members, table["syst_alleras"], channels=channels))
            entries.append(LnN(nuisance("CMS_eff_t_vsJet", "tau_mc", era), members, table["era"], channels=channels))
    for channel, sizes in study_sizes.LEPTONS[era].items():
        if "muon_stat" in sizes:
            entries.append(LnN(nuisance("CMS_eff_m_stat", "muon_id_stat", era), mc, sizes["muon_stat"], channels=(channel,)))
        if "electron_reco" in sizes:
            entries.append(LnN(nuisance("CMS_eff_e_reco", "electron_reco", era), mc, sizes["electron_reco"], channels=(channel,)))
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


def lnn(processes: tuple[Process, ...], jet_fakes_output: str, era: str, borrowed: tuple[str, ...] = (), study: Study | None = None, dy_correction: str = "none") -> tuple[LnN, ...]:
    """Normalisation uncertainties; `borrowed` are the processes made of samples borrowed from another era; `study`
    adds the lnN of the switch study_uncertainties (study_lnn), whose POG vsJet components replace the flat
    eff_t_vsJet; with the DY table (dy_correction njets_btag) its uncertainty (dy_table_lnn)."""
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
        *(() if study else (LnN(tau("eff_t_vsJet_$CHANNEL"), mc, 1.02, channels=LT_CHANNELS), LnN(tau("eff_t_vsJet_$CHANNEL"), mc, 1.0404, channels=("tt",)))),
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
        *(study_lnn(processes, jet_fakes_output, era, study) if study else ()),
        *((dy_table_lnn(processes, era),) if dy_correction == "njets_btag" else ()),
    ]
    return tuple(e for e in entries if e.processes)
