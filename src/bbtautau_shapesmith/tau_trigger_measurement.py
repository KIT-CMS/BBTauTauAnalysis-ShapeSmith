"""The efficiency of the tau leg of the di-tau trigger in data for the tt embedding (v7 spec E33, variant A;
docs v7_tt_emb_trigger_2026-10-16 sections 12.2, 12.5), measured with mu tau_h tag and probe on the tau-ID production
P1 and run by the core measurement shapesmith.measurements.tau_trigger.

Tag: the single-muon trigger of the tau-ID measurement of the era (triggers.TRIGGER[era]["tau_id"]) with the muon in
the plateau of the monitoring path, pT_1 > 24 / 29 / 28 GeV (2016 / 2017 / 2018), iso < 0.15, |eta| < 2.1 (E31).
Probe: the tau_h with pT >= 40 GeV, |eta| < 2.1 (the tt acceptance), decay mode 0, 1, 10, 11, VSmu Tight, VSe VVLoose,
in its isolation class (switch wp_class): Medium (the signal region of tt) or VVVLoose and not Medium (the
application region of the fake factors, whose embedded taus are subtracted). Event: opposite sign (QCD from same
sign), mT < 65 GeV, the lepton vetoes, the 2018 HEM veto (E29).
Pass: the monitoring mu tau path of the era fired with the probe matched to its tau-leg filters (the CROWN columns
trg_mon_*_2 of C13/12.4), mirroring the tt OR leg by leg (12.2): 2016 MediumIso (data also MediumCombinedIso) with the
matched object above 35 GeV; 2017 Tight35_TightID or (Medium35_TightID or Tight35) above 40 GeV, L1 >= 32 GeV; 2018
data before run 317509 (HPS_FIRST_RUN) the 2017 paths without the L1 cut, the data from run 317509 on, MC and embedding
the HPS path. The split follows the di-tau trigger of the data, which is HPS only from run 317509, not the monitoring
path: the HPS mu tau monitoring path fires from the first 2018 run (315357), and legs or HPS raised the pass fraction of
2018A data from 0.555 to 0.581 (smrun2_test_tm_v7a). MC and embedding have the HPS path only; the embedding keeps the
run number of its data event, so the split asks for is_data.
Bins: per decay mode (0, 1, 10, 11 and all, dm -1; with dm1011 10 and 11 together) and pT ([40, 45, 50, 60, 80, 120,
inf) or the edges of TRIGGER_PT_EDGES for the era and class, from `measure --suggest-binning`), each a pass and a fail
category in m_vis with ten equal-data bins in 30-160 GeV from the data of the bin (pass + fail), so that the embedded
signal has one shape in both.
Processes: the embedded genuine taus (the signal; their energy scale from the measured payload of 5.4, switch
emb_es_payload, applied as the template variation emb_es with its +-1 sigma as a shape nuisance; its ttbar
contamination as in the tau-ID measurement), ZL, ZJ, TTL, TTJ, STL, STJ, VVL, VVJ, W from MC, QCD = same-sign data
minus all of them; the genuine MC taus ZTT (and TTT) are auxiliary, ZTT gives eps_MC. Normalisation lnN W 20 %, TT 10 %,
VV/ST 10 %, ZL 10 %, ZJ 20 %, QCD 20 %; the fake trigger efficiency +-10 % of the pass yield of ZJ, TTJ, STJ, VVJ, W, QCD.
Weights: those of the tau-ID measurement (tau_id_measurement.mc_weights / embedding_weights) with the Medium tau-ID SF
in the Medium class and none in the anti-isolated one.
Variants (switch variant; merged as uncertainties, 12.2): signal_mc (the MC genuine taus as the signal with their own
pass and fail shapes, the full change), tag_pt (the tag muon 3 GeV harder), mt30 (mT < 30), monitor (2016 without the
35 GeV object cut, 2017 without the L1 cut; half of each change). closure (12.7.2, not merged): the nominal method on
MC, the MC genuine taus as the signal and the data replaced by the sum of the processes; the fit must give eps_MC.
The signal shapes are the ES nuisance (a column variation) and the ttbar contamination (a template shift, an estimator
output). eps_EMB^mon (the embedded taus counted with the monitoring flags) only in 2016 and 2017
(EMBEDDING_MONITOR_YEARS): the 2018 mu tau embedding has no monitoring-path decisions (the HPS paths fire in none of
1.34M events of 2018A, its tau trigger objects carry only the bits 1, 22 and 23), so its pass categories stay empty and
the measurement has no monitor process there; the fit does not need them (the signal is one template, pass + fail).
Plots: eps(pT) with the POG v15 eff_data and eff_mc (switch reference_payload) and the KIT embedding file's eff_data
and eff_emb (kit_reference_payload). --merge needs a nominal measurement of both classes (CROWN books both).
"""
from __future__ import annotations

import logging
from pathlib import Path

from shapesmith.config import RunConfig
from shapesmith.measurements.tau_trigger import Bin, Reference, TauTriggerMeasurement
from shapesmith.model import Analysis, AnalysisError, Category, Channel, ColumnVariation, DataMinus, EqualData, Process, Region, Selection, TemplateShift, Variable

from bbtautau_shapesmith.constants import EVENT_KEY, LUMI_PB, YEAR, check_era
from bbtautau_shapesmith.cuts import SAME_SIGN, era_cuts
from bbtautau_shapesmith.samples import sample_database, samples
from bbtautau_shapesmith.switches import TauTriggerSwitches, parse
from bbtautau_shapesmith.tau_id_measurement import GENUINE, MUON_ETA, PART_CUTS_MT, SPLITS_MT, VS_MU_WP, embedding_weights, mc_weights
from bbtautau_shapesmith.tau_id_systematics import ERA_TAG
from bbtautau_shapesmith.triggers import TRIGGER
from bbtautau_shapesmith.weights import TOP_PT

logger = logging.getLogger(__name__)

WP_CLASSES = {
    "Medium": "(id_tau_vsJet_Medium_2 > 0.5)",
    "VVVLoose_not_Medium": "((id_tau_vsJet_VVVLoose_2 > 0.5) & (id_tau_vsJet_Medium_2 < 0.5))",
}
VS_ELE_WP = "VVLoose"
TAG_PT = {"2016": 24.0, "2017": 29.0, "2018": 28.0}  # the muon of the monitoring path in its plateau (12.2)
TAG_PT_VARIATION = 3.0  # GeV, the variant tag_pt
PROBE = {"tau_pt": "(pt_2 >= 40)", "tau_eta": "(abs(eta_2) < 2.1)"}
PT_EDGES = (40.0, 45.0, 50.0, 60.0, 80.0, 120.0)  # the last bin open above
# era -> wp_class -> decay-mode group -> pT edges (the last bin open above), entered from `measure --suggest-binning`
# on smrun2_tau_id_v1 (statistical error of eps_data at most 0.03, counted with mT < 30 GeV and 40 < m_vis < 90 GeV).
# The anti-isolated class reaches it in no bin, so each of its decay-mode groups is one bin from 40 GeV.
SINGLE_BIN = {group: (40.0,) for group in ("DMall", "DM0", "DM1", "DM10", "DM11", "DM1011")}
TRIGGER_PT_EDGES: dict[str, dict[str, dict[str, tuple[float, ...]]]] = {
    "2016preVFP": {
        "Medium": {"DMall": PT_EDGES, "DM0": (40.0, 50.0, 80.0), "DM1": (40.0, 45.0, 50.0, 60.0, 80.0), "DM10": (40.0, 45.0, 50.0, 60.0), "DM11": (40.0, 60.0)},
        "VVVLoose_not_Medium": SINGLE_BIN,
    },
    "2016postVFP": {
        "Medium": {"DMall": PT_EDGES, "DM0": (40.0, 50.0, 80.0), "DM1": (40.0, 45.0, 50.0, 60.0, 80.0), "DM10": (40.0, 45.0, 50.0, 60.0), "DM11": (40.0, 45.0, 80.0)},
        "VVVLoose_not_Medium": SINGLE_BIN,
    },
    "2017": {
        "Medium": {"DMall": PT_EDGES, "DM0": (40.0, 45.0), "DM1": (40.0, 45.0, 50.0, 60.0, 80.0), "DM10": (40.0, 45.0, 50.0, 80.0), "DM11": (40.0, 50.0)},
        "VVVLoose_not_Medium": SINGLE_BIN,
    },
    "2018": {
        "Medium": {"DMall": PT_EDGES, "DM0": (40.0, 45.0, 80.0), "DM1": PT_EDGES, "DM10": (40.0, 45.0, 50.0, 80.0, 120.0), "DM11": (40.0, 50.0)},
        "VVVLoose_not_Medium": SINGLE_BIN,
    },
}
COUNTING_WINDOW = (40.0, 90.0)  # GeV of m_vis, the counting cross-check (with mT < 30 GeV) and --suggest-binning
MAX_ERROR = 0.03  # the largest statistical error of eps_data per bin (absolute), --suggest-binning
M_VIS = EqualData(10, 30.0, 160.0)  # as tau_id_binning.M_VIS_BINNING, from the data of the bin (pass + fail)
EMBEDDED = "EMB"
GENUINE_MC = "ZTT"  # the MC genuine taus: eps_MC, the signal of the variant signal_mc
ES_TEMPLATE = "emb_es"
ES_REFERENCE = {"pt": 50.0, "eta": 0.0, "wp": "Medium", "wp_VSe": VS_ELE_WP}  # where the dm-binned ES of 5.4 is read

# the monitoring flags of the CROWN tau-ID production (C13, 12.4) and the matched trigger object
MONITOR_FLAGS = {
    "2016": {"iso": "trg_mon_mu19tau32_mediso_2", "combined_iso": "trg_mon_mu19tau32_medcombiso_2"},
    "2017": {"tight_tightid": "trg_mon_mu24tau35_tightiso_tightid_2", "medium_tightid": "trg_mon_mu24tau35_mediumiso_tightid_2", "tight": "trg_mon_mu24tau35_tightiso_2"},
}
MONITOR_FLAGS["2018"] = {**MONITOR_FLAGS["2017"], "hps": "trg_mon_mu24tau35_mediumiso_hps_2"}
TRIGGER_OBJECT = {"pt": "trgobj_pt_2", "l1pt": "trgobj_l1pt_2"}
HPS_FIRST_RUN = 317509  # 2018 data: the di-tau trigger is HPS from this run on, before it the 2017 paths (12.2)
EMBEDDING_MONITOR_YEARS = ("2016", "2017")  # eps_EMB^mon: the 2018 mu tau embedding has no monitoring-path decisions
VARIANTS = ("nominal", "signal_mc", "tag_pt", "mt30", "monitor", "closure")
MC_SIGNAL_VARIANTS = ("signal_mc", "closure")  # the MC genuine taus are the signal
VARIANT_WEIGHTS = {"signal_mc": 1.0, "tag_pt": 0.5, "mt30": 0.5, "monitor": 0.5}  # the fraction of a variant's change taken as uncertainty

NORMALISATIONS = {
    "CMS_tauTrg_norm_W": (("W",), 1.2),
    "CMS_tauTrg_norm_TT": (("TTL", "TTJ", "TTT"), 1.1),
    "CMS_tauTrg_norm_VVST": (("VVL", "VVJ", "STL", "STJ"), 1.1),
    "CMS_tauTrg_norm_ZL": (("ZL",), 1.1),
    "CMS_tauTrg_norm_ZJ": (("ZJ",), 1.2),
    "CMS_tauTrg_norm_QCD": (("QCD",), 1.2),
}
FAKES = ("ZJ", "TTJ", "STJ", "VVJ", "W", "QCD")
# the plotted references (12.5): the POG file of the era and the KIT embedding file (DeepTau2017v2p1; its eff_mc is the
# efficiency of the embedded taus, eff_emb)
POG_LABELS = {"eff_data": "POG v15 eff_data", "eff_mc": "POG v15 eff_mc"}
KIT_CORRECTION, KIT_LABELS = "tauTriggerSF", {"eff_data": "KIT eff_data (2017v2p1)", "eff_mc": "KIT eff_emb"}


def references(switches: TauTriggerSwitches) -> tuple[Reference, ...]:
    pog = (Reference(switches.reference_payload, labels=POG_LABELS),) if switches.reference_payload else ()
    kit = (Reference(switches.kit_reference_payload, KIT_CORRECTION, KIT_LABELS),) if switches.kit_reference_payload else ()
    return pog + kit


def fired(flag: str) -> str:
    return f"({flag} > 0.5)"


def pass_cut(year: str, variant: str = "nominal") -> str:
    """The probe passes the monitoring leg (12.2); in 2016 the data take MediumIso or MediumCombinedIso, MC and
    embedding MediumIso only (E7, D9); in 2018 the data before run HPS_FIRST_RUN the 2017 legs without the L1 cut, the
    rest (later data, MC, embedding) the HPS path."""
    flags, obj = MONITOR_FLAGS[year], TRIGGER_OBJECT
    if year == "2016":
        data = f"((is_data > 0.5) & ({fired(flags['iso'])} | {fired(flags['combined_iso'])}))"
        simulated = f"((is_data < 0.5) & {fired(flags['iso'])})"
        paths = f"({data} | {simulated})"
        return paths if variant == "monitor" else f"({paths} & ({obj['pt']} > 35))"
    legs = f"({fired(flags['tight_tightid'])} | (({fired(flags['medium_tightid'])} | {fired(flags['tight'])}) & ({obj['pt']} > 40)))"
    if year == "2017":
        return legs if variant == "monitor" else f"({legs} & ({obj['l1pt']} >= 32))"
    before_hps = f"((is_data > 0.5) & (run < {HPS_FIRST_RUN}))"
    return f"(({before_hps} & {legs}) | (~{before_hps} & {fired(flags['hps'])}))"


def decay_mode_groups(dm1011: bool) -> dict[str, tuple[int, ...]]:
    groups = {"DMall": (-1,), "DM0": (0,), "DM1": (1,)}
    return {**groups, "DM1011": (10, 11)} if dm1011 else {**groups, "DM10": (10,), "DM11": (11,)}


def bin_name(group: str, low: float, high: float | None) -> str:
    return f"{group}_PT{low:g}" + ("" if high is None else f"_{high:g}")


def trigger_bins(era: str, wp_class: str, dm1011: bool) -> tuple[Bin, ...]:
    edges_by_group = TRIGGER_PT_EDGES.get(era, {}).get(wp_class, {})
    result = []
    for group, modes in decay_mode_groups(dm1011).items():
        edges = edges_by_group.get(group, PT_EDGES)
        for low, high in zip(edges, (*edges[1:], None)):
            result.append(Bin(bin_name(group, low, high), modes, (low, high)))
    return tuple(result)


def bin_cut(trigger_bin: Bin) -> str:
    modes = trigger_bin.decay_modes
    dm = "(tau_decaymode_2 >= 0)" if modes == (-1,) else "(" + " | ".join(f"(tau_decaymode_2 == {m})" for m in modes) + ")"
    low, high = trigger_bin.pt
    return f"{dm} & (pt_2 >= {low:g})" + ("" if high is None else f" & (pt_2 < {high:g})")


def categories(bins: tuple[Bin, ...], year: str, variant: str) -> tuple[Category, ...]:
    passing = pass_cut(year, variant)
    result = []
    for b in bins:
        variable = Variable("m_vis", "m_vis", EqualData(M_VIS.n_bins, M_VIS.low, M_VIS.high, cut=bin_cut(b)))
        result += [Category(b.category("pass"), f"{bin_cut(b)} & {passing}", variable), Category(b.category("fail"), f"{bin_cut(b)} & ~{passing}", variable)]
    return tuple(result)


def cuts(era: str, wp_class: str, variant: str = "nominal") -> dict[str, str]:
    year = YEAR[era]
    tag_pt = TAG_PT[year] + (TAG_PT_VARIATION if variant == "tag_pt" else 0.0)
    return {
        "os": "((q_1 * q_2) < 0)",
        "tau_decay_mode": "((tau_decaymode_2 == 0) | (tau_decaymode_2 == 1) | (tau_decaymode_2 == 10) | (tau_decaymode_2 == 11))",
        **PROBE,
        "extraelec_veto": "(extraelec_veto < 0.5)",
        "extramuon_veto": "(extramuon_veto < 0.5)",
        "dilepton_veto": "(dilepton_veto < 0.5)",
        "against_muon": f"(id_tau_vsMu_{VS_MU_WP}_2 > 0.5)",
        "against_electron": f"(id_tau_vsEle_{VS_ELE_WP}_2 > 0.5)",
        "tau_iso": WP_CLASSES[wp_class],
        "muon_iso": "(iso_1 < 0.15)",
        "mt_cut": "(mt_1 < 30)" if variant == "mt30" else "(mt_1 < 65)",
        "muon_eta": MUON_ETA,
        "trg_selection": f"(pt_1 > {tag_pt:g}) & {TRIGGER[era]['tau_id'].cut}",
        **era_cuts(era),
    }


def skim_cuts(era: str) -> dict[str, str]:
    """Both charges, both isolation classes, the loosest tag and mT of every variant."""
    loose = {**cuts(era, "VVVLoose_not_Medium"), "tau_iso": "(id_tau_vsJet_VVVLoose_2 > 0.5)"}
    return {name: expr for name, expr in loose.items() if name != "os"}


def weights(wp_class: str, era: str) -> dict[str, str]:
    """The MC weights of the tau-ID measurement; no tau-ID SF in the anti-isolated class (none exists for it)."""
    result = mc_weights("Medium", VS_ELE_WP, era)
    return result if wp_class == "Medium" else {**result, "tau_id": "1.0"}


def processes(wp_class: str, era: str, variant: str) -> tuple[Process, ...]:
    mc = weights(wp_class, era)

    def simulated(name: str, group: str, cut: str | None, role: str = "background") -> Process:
        extra = dict(TOP_PT) if group == "TT" else {}
        return Process(name, group, role, group, Selection(cuts={"genmatch": cut} if cut else {}, weights={**mc, **extra}))

    signal_mc = variant in MC_SIGNAL_VARIANTS
    result = [Process("data", "data", "data", "data")]
    if not signal_mc:
        result.append(Process(EMBEDDED, EMBEDDED, "background", EMBEDDED, Selection(cuts={"genmatch": GENUINE}, weights=embedding_weights("mt", era))))
    result.append(simulated(GENUINE_MC, "DY", GENUINE, "background" if signal_mc else "auxiliary"))
    for group, parts in SPLITS_MT.items():
        for name, part in parts.items():
            role = "auxiliary" if part == "T" and not signal_mc else "background"
            result.append(simulated(name, group, PART_CUTS_MT[part], role))
    result.append(simulated("W", "W", None))
    return tuple(result)


def _scale_expression(factors: dict[int, float]) -> str:
    return "(" + " + ".join(f"(tau_decaymode_2 == {dm}) * {factor!r}" for dm, factor in sorted(factors.items())) + ")"


def es_variation(name: str, factors: dict[int, float]) -> ColumnVariation:
    """The embedded tau scaled by s(DM), propagated to m_vis, the MET and mt_1 (as the ES grid of the tau-ID
    measurement)."""
    s = _scale_expression(factors)
    met_x = f"(met * cos(metphi) + (1 - {s}) * pt_2 * cos(phi_2))"
    met_y = f"(met * sin(metphi) + (1 - {s}) * pt_2 * sin(phi_2))"
    met, metphi = f"sqrt({met_x} ** 2 + {met_y} ** 2)", f"arctan2({met_y}, {met_x})"
    derived = {
        "pt_2": f"pt_2 * {s}",
        "mass_2": f"mass_2 * {s}",
        "m_vis": f"sqrt(mass_1 ** 2 + ({s} * mass_2) ** 2 + {s} * (m_vis ** 2 - mass_1 ** 2 - mass_2 ** 2))",
        "met": met,
        "metphi": metphi,
        "mt_1": f"sqrt(2 * pt_1 * {met} * (1 - cos(phi_1 - {metphi})))",
    }
    return ColumnVariation(name, derived=derived, applies_to=("embedding",), regions=("nominal",))


def es_factors(payload: str) -> dict[str, dict[int, float]]:
    """syst (nom, up, down) -> decay mode -> the ES factor of the embedded taus from the payload of 5.4
    (tau_energy_scale_dm_binned at ES_REFERENCE; the Medium measurement serves both isolation classes)."""
    import correctionlib

    evaluator = correctionlib.CorrectionSet.from_file(payload)["tau_energy_scale_dm_binned"]
    reference = ES_REFERENCE
    return {
        syst: {dm: float(evaluator.evaluate(reference["pt"], reference["eta"], dm, 5, "DeepTau2018v2p5", reference["wp"], reference["wp_VSe"], syst)) for dm in (0, 1, 10, 11)}
        for syst in ("nom", "up", "down")
    }


def es_nuisance(era: str) -> str:
    return f"CMS_tauTrg_emb_es_{ERA_TAG[era]}"


def channel(era: str, switches: TauTriggerSwitches, channel_samples, bins: tuple[Bin, ...]) -> Channel:
    year = YEAR[era]
    table = processes(switches.wp_class, era, switches.variant)
    groups = {p.group for p in table}
    backgrounds = tuple(p.name for p in table if p.role == "background")
    variations, estimators = (), [DataMinus("QCD", "same_sign", backgrounds, clip_negative=True)]
    if switches.variant not in MC_SIGNAL_VARIANTS:
        estimators.append(TemplateShift(f"CMS_emb_ttbar_contamination_{ERA_TAG[era]}", EMBEDDED, "TTT", 0.1))
        if switches.emb_es_payload:
            factors = es_factors(switches.emb_es_payload)
            variations = (es_variation(ES_TEMPLATE, factors["nom"]), es_variation(f"{es_nuisance(era)}Up", factors["up"]), es_variation(f"{es_nuisance(era)}Down", factors["down"]))
        else:
            logger.warning("tau_trigger: no emb_es_payload, the embedded taus carry no energy scale (the measured payload of 5.4 comes first)")
    return Channel(
        name="mt",
        samples=tuple(s for s in channel_samples if s.group in groups),
        skim=skim_cuts(era),
        cuts=cuts(era, switches.wp_class, switches.variant),
        processes=table,
        regions=(Region("same_sign", replace_cuts={"os": SAME_SIGN}), Region("counting", replace_cuts={"mt_cut": "(mt_1 < 30)"}),
                 Region("counting_same_sign", replace_cuts={"mt_cut": "(mt_1 < 30)", "os": SAME_SIGN})),
        categories=categories(bins, year, switches.variant),
        variations=variations,
        estimators=tuple(estimators),
        keep_columns=("event", "run", "lumi"),
    )


def measurement(era: str, switches: TauTriggerSwitches, bins: tuple[Bin, ...]) -> TauTriggerMeasurement:
    year = YEAR[era]
    signal_mc = switches.variant in MC_SIGNAL_VARIANTS
    table = processes(switches.wp_class, era, switches.variant)
    names = {p.name for p in table if p.role == "background"}
    templated = switches.emb_es_payload is not None and not signal_mc
    return TauTriggerMeasurement(
        wp_class=switches.wp_class,
        bins=bins,
        signal=GENUINE_MC if signal_mc else EMBEDDED,
        variant=switches.variant,
        systematic_variants={name: weight for name, weight in VARIANT_WEIGHTS.items() if name != "monitor" or year != "2018"},
        signal_template=ES_TEMPLATE if templated else None,
        signal_shapes=((es_nuisance(era),) if templated else ()) + (() if signal_mc else (f"CMS_emb_ttbar_contamination_{ERA_TAG[era]}",)),
        normalisations={name: (tuple(p for p in group if p in names), kappa) for name, (group, kappa) in NORMALISATIONS.items()},
        fake_processes=FAKES,
        mc_process=GENUINE_MC,
        monitor_process=EMBEDDED if not signal_mc and year in EMBEDDING_MONITOR_YEARS else None,
        counting_regions=("counting", "counting_same_sign"),
        counting_window=COUNTING_WINDOW,
        # --suggest-binning (12.2): merge from the top until the statistical error of eps_data is at most 0.03, counted
        # where the counting cross-check counts (the whole mT < 65 GeV region inflates it by the background subtraction)
        suggest_regions=("counting", "counting_same_sign"),
        suggest_window=COUNTING_WINDOW,
        max_error=MAX_ERROR,
        relative_max_error=False,
        qcd_subtract=tuple(p.name for p in table if p.role == "background"),
        references=references(switches),
        wp_classes=tuple(WP_CLASSES),  # CROWN books both classes (trg_effdata_emb_{Medium,AntiMedium}_<leg>)
        split_signal_shapes=switches.variant == "signal_mc",
        pseudo_data=switches.variant == "closure",
    )


def build(config: RunConfig) -> Analysis:
    era = check_era(config.era)
    switches = parse(TauTriggerSwitches, config.switches)
    if list(config.channels) != ["mt"]:
        raise AnalysisError("the tau-leg trigger measurement runs in the channel mt only")
    if switches.variant == "monitor" and YEAR[era] == "2018":
        raise AnalysisError("the variant monitor applies to 2016 (object pT > 35 GeV) and 2017 (L1 >= 32 GeV) only")
    if switches.emb_es_payload is not None and not Path(switches.emb_es_payload).exists():
        raise AnalysisError(f"emb_es_payload {switches.emb_es_payload} does not exist")
    if YEAR[era] not in EMBEDDING_MONITOR_YEARS and switches.variant not in MC_SIGNAL_VARIANTS:
        logger.info(f"tau_trigger {era}: the mu tau embedding has no monitoring-path decisions, eps_EMB^mon is not measured")
    bins = trigger_bins(era, switches.wp_class, switches.dm1011)
    by_channel = samples(sample_database(config), switches.sample_lists, ["mt"], era)
    return Analysis(
        name=f"tau_trigger_embedding_{era}",
        era=era,
        lumi_pb=LUMI_PB[era],
        signal=None,
        channels={"mt": channel(era, switches, by_channel["mt"], bins)},
        measurement=measurement(era, switches, bins),
        event_key=EVENT_KEY,
    )

