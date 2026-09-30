"""The tau-ID scale factor and energy scale measurement of the embedded taus, 2018 (successor of smhtt_ul tauID_SFs_dev).

Tag and probe in mt: the muon tags a Z->tautau->mu tau_h event, the tau is probed in nine decay-mode (and pT)
categories; the Z->mumu control region in mm shares the Z normalisation. Selections and weights are those of
smhtt_ul config/shapes (channel_selection and process_selection with special "TauID_ES",
tauid_measurement_binning), with the column names of the bbtautau measurement configuration. The switches vsjet_wp
and vsele_wp choose the working points; one skim serves all four combinations.

The embedded signal carries no tau ID or ES correction. Its energy scale is a grid of column variations derived from
the nominal columns: the tau four-momentum scaled by s (pt_2, mass_2, m_vis), the MET corrected by (1 - s) times the
tau pT, and mt_1 from that MET.

The MC is normalised to the analysis luminosity, 59.83 fb-1 (constants.LUMI_PB); the predecessor used 59.56 fb-1.
The embedded samples are data-driven and unaffected.
"""
from __future__ import annotations

from shapesmith.config import RunConfig
from shapesmith.measurements.tau_id_es import TauIdEsMeasurement
from shapesmith.measurements.tau_id_es.grid import CATEGORIES, CONTROL_CATEGORY, SIGNAL, factor, grid, grid_name
from shapesmith.model import Analysis, Category, Channel, ColumnVariation, DataMinus, Process, Region, Sample, Selection, TemplateShift, Variable

from bbtautau_shapesmith.constants import ERA, LUMI_PB
from bbtautau_shapesmith.samples import sample_database, samples
from bbtautau_shapesmith.switches import TauIdSwitches, parse
from bbtautau_shapesmith.tau_id_binning import CONTROL_REGION_EDGES, M_VIS_EDGES
from bbtautau_shapesmith.tau_id_systematics import ERA_TAG, mc_variations

ES_GRID = grid(-200, 200, 2)  # +-20 % in steps of 0.2 %, as the predecessor's 2018 measurement
VS_MU_WP = "Tight"
MEASURED_WPS = {"vsjet": ("Medium", "Tight"), "vsele": ("VVLoose", "Tight")}
MUON_EMBEDDED = "MUEMB"
TRIGGER = "((trg_single_mu27 > 0.5) | (trg_single_mu24 > 0.5))"
TRIGGER_WEIGHT = "((pt_1 >= 25) & (pt_1 < 28)) * trg_wgt_single_mu24 + (pt_1 > 28) * trg_wgt_single_mu27"
SAME_SIGN = "((q_1 * q_2) > 0)"

DECAY_MODE_CUTS = {"DM0": "(tau_decaymode_2 == 0)", "DM1": "(tau_decaymode_2 == 1)", "DM1011": "((tau_decaymode_2 == 10) | (tau_decaymode_2 == 11))"}
PT_CUTS = {"": "(pt_2 >= 20)", "_PT20_40": "(pt_2 >= 20) & (pt_2 < 40)", "_PT40_200": "(pt_2 >= 40) & (pt_2 <= 200)"}

# gen-match parts in mt: T = genuine tau_h with the muon from a tau decay (the embedded events), J = jet -> tau_h fakes
GENUINE = "((gen_match_1 == 4) & (gen_match_2 == 5))"
JET_FAKE = "(gen_match_2 == 6)"
SPLITS_MT = {"DY": {"ZL": "L", "ZJ": "J"}, "TT": {"TTT": "T", "TTL": "L", "TTJ": "J"}, "ST": {"STL": "L", "STJ": "J"}, "VV": {"VVL": "L", "VVJ": "J"}}
PART_CUTS_MT = {"T": GENUINE, "L": f"(~{GENUINE} & ~{JET_FAKE})", "J": JET_FAKE}


def mt_cuts(vsjet: str, vsele: str) -> dict[str, str]:
    return {
        "os": "((q_1 * q_2) < 0)",
        "tau_decay_mode": "((tau_decaymode_2 == 0) | (tau_decaymode_2 == 1) | (tau_decaymode_2 == 10) | (tau_decaymode_2 == 11))",
        "tau_eta": "(abs(eta_2) < 2.5)",
        "extraelec_veto": "(extraelec_veto < 0.5)",
        "extramuon_veto": "(extramuon_veto < 0.5)",
        "dilepton_veto": "(dilepton_veto < 0.5)",
        "against_muon": f"(id_tau_vsMu_{VS_MU_WP}_2 > 0.5)",
        "against_electron": f"(id_tau_vsEle_{vsele}_2 > 0.5)",
        "tau_iso": f"(id_tau_vsJet_{vsjet}_2 > 0.5)",
        "muon_iso": "(iso_1 < 0.15)",
        "mt_cut": "(mt_1 < 65)",
        "trg_selection": f"(pt_2 > 20) & (pt_1 >= 25) & {TRIGGER}",
    }


def mm_cuts() -> dict[str, str]:
    return {
        "os": "((q_1 * q_2) < 0)",
        "m_vis": "(m_vis > 70) & (m_vis < 110)",
        "muon_iso": "(iso_1 < 0.15) & (iso_2 < 0.15)",
        "trg_selection": f"(pt_2 > 15) & (pt_1 >= 25) & {TRIGGER}",
    }


def skim_cuts(cuts: dict[str, str]) -> dict[str, str]:
    """Both charges (for the same-sign QCD estimate) and the loosest measured tau ID, so that one skim serves every
    working-point combination."""
    loose = {"tau_iso": f"(id_tau_vsJet_{MEASURED_WPS['vsjet'][0]}_2 > 0.5)", "against_electron": f"(id_tau_vsEle_{MEASURED_WPS['vsele'][0]}_2 > 0.5)"}
    return {name: loose.get(name, expr) for name, expr in cuts.items() if name != "os"}


def working_point_regions() -> tuple[Region, ...]:
    """The selection of every measured working-point combination as a region: its columns are in the skim, so one
    skim serves all four, and any of them can be filled from it (`hist --regions`)."""
    return tuple(
        Region(f"wp_{vsjet}_{vsele}", replace_cuts={"tau_iso": f"(id_tau_vsJet_{vsjet}_2 > 0.5)", "against_electron": f"(id_tau_vsEle_{vsele}_2 > 0.5)"},
               replace_weights={"tau_id": mc_weights(vsjet, vsele)["tau_id"], "vs_ele": mc_weights(vsjet, vsele)["vs_ele"]})
        for vsjet in MEASURED_WPS["vsjet"] for vsele in MEASURED_WPS["vsele"]
    )


def categories(vsjet: str, vsele: str) -> tuple[Category, ...]:
    edges = M_VIS_EDGES[vsjet, vsele]
    result = []
    for dm, dm_cut in DECAY_MODE_CUTS.items():
        for pt, pt_cut in PT_CUTS.items():
            name = f"{dm}{pt}"
            result.append(Category(name, f"{dm_cut} & {pt_cut}", Variable("m_vis", "m_vis", edges[name])))
    return tuple(sorted(result, key=lambda c: list(CATEGORIES).index(c.name)))


def es_variation(shift: int) -> ColumnVariation:
    """One grid point: the embedded tau scaled by s = 1 + shift / 1000, propagated to m_vis, the MET and mt_1."""
    s, recoil = factor(shift), -shift / 1000  # the MET gains (1 - s) times the tau pT
    met_x = f"(met * cos(metphi) + {recoil!r} * pt_2 * cos(phi_2))"
    met_y = f"(met * sin(metphi) + {recoil!r} * pt_2 * sin(phi_2))"
    met, metphi = f"sqrt({met_x} ** 2 + {met_y} ** 2)", f"arctan2({met_y}, {met_x})"
    derived = {
        "pt_2": f"pt_2 * {s!r}",
        "mass_2": f"mass_2 * {s!r}",
        "m_vis": f"sqrt(mass_1 ** 2 + ({s!r} * mass_2) ** 2 + {s!r} * (m_vis ** 2 - mass_1 ** 2 - mass_2 ** 2))",
        "met": met,
        "metphi": metphi,
        "mt_1": f"sqrt(2 * pt_1 * {met} * (1 - cos(phi_1 - {metphi})))",
    }
    return ColumnVariation(grid_name(shift), derived=derived, applies_to=("embedding",), regions=("nominal",))


def mc_weights(vsjet: str, vsele: str) -> dict[str, str]:
    return {
        "puweight": "puweight",
        "id": "id_wgt_mu_1",
        "iso": "iso_wgt_mu_1",
        "tau_id": f"((gen_match_2 == 5) * id_wgt_tau_vsJet_{vsjet}_2 + (gen_match_2 != 5))",
        "vs_mu": f"id_wgt_tau_vsMu_{VS_MU_WP}_2",
        "vs_ele": f"id_wgt_tau_vsEle_{vsele}_2",
        "trigger": TRIGGER_WEIGHT,
    }


def mm_weights() -> dict[str, str]:
    return {"puweight": "puweight", "id": "id_wgt_mu_1 * id_wgt_mu_2", "iso": "iso_wgt_mu_1 * iso_wgt_mu_2", "trigger": TRIGGER_WEIGHT}


def embedding_weights(channel: str) -> dict[str, str]:
    """The embedding selection weights and the muon SFs (both legs in mm); no tau ID or ES correction."""
    muons = {"mt": {"emb_id": "id_wgt_mu_1", "emb_iso": "iso_wgt_mu_1"}, "mm": {"emb_id": "id_wgt_mu_1 * id_wgt_mu_2", "emb_iso": "iso_wgt_mu_1 * iso_wgt_mu_2"}}[channel]
    return {"emb_genweight": "emb_genweight", "emb_selection": "emb_idsel_wgt_1 * emb_idsel_wgt_2 * emb_triggersel_wgt", **muons, "emb_trigger": TRIGGER_WEIGHT}


def _mc(name: str, group: str, weights: dict[str, str], cut: str | None = None, role: str = "background", jet_fake: bool = False) -> Process:
    extra = {"top_pt": "topPtReweightWeight"} if group == "TT" else {}
    if jet_fake:
        extra["jet_fake"] = "1.0"  # replaced by the CMS_fake_j variation
    return Process(name, group, role, group, Selection(cuts={"genmatch": cut} if cut else {}, weights={**weights, **extra}))


def mt_processes(vsjet: str, vsele: str) -> tuple[Process, ...]:
    """Data, the embedded signal, the MC lepton and jet fakes, W; genuine ttbar is the template of the embedding ttbar
    contamination."""
    weights = mc_weights(vsjet, vsele)
    result = [Process("data", "data", "data", "data"), Process(SIGNAL, "EMB", "background", "EMB", Selection(cuts={"genmatch": GENUINE}, weights=embedding_weights("mt")))]
    for group, parts in SPLITS_MT.items():
        result += [_mc(name, group, weights, PART_CUTS_MT[part], "auxiliary" if part == "T" else "background", part == "J") for name, part in parts.items()]
    result.append(_mc("W", "W", weights, jet_fake=True))
    return tuple(result)


def mm_processes() -> tuple[Process, ...]:
    """Data, the embedded muons (both genuine prompt muons), W and the MC ttbar and diboson without Z->tautau->mumu."""
    weights, no_tautau = mm_weights(), "~((gen_match_1 == 4) & (gen_match_2 == 4))"
    return (
        Process("data", "data", "data", "data"),
        Process(MUON_EMBEDDED, MUON_EMBEDDED, "background", "EMB", Selection(cuts={"genmatch": "(gen_match_1 == 2) & (gen_match_2 == 2)"}, weights=embedding_weights("mm"))),
        _mc("W", "W", weights), _mc("TTL", "TT", weights, no_tautau), _mc("VVL", "VV", weights, no_tautau),
    )


def _channel(name: str, cuts: dict[str, str], processes: tuple[Process, ...], channel_samples: tuple[Sample, ...], regions: tuple[Region, ...] = (), **fields) -> Channel:
    groups = {p.group for p in processes}
    return Channel(
        name=name,
        samples=tuple(s for s in channel_samples if s.group in groups),
        skim=skim_cuts(cuts),
        cuts=cuts,
        processes=processes,
        regions=(Region("same_sign", replace_cuts={"os": SAME_SIGN}), *regions),
        keep_columns=("event", "run", "lumi"),
        **fields,
    )


def mt_channel(switches: TauIdSwitches, channel_samples: tuple[Sample, ...]) -> Channel:
    processes = mt_processes(switches.vsjet_wp, switches.vsele_wp)
    subtract = tuple(p.name for p in processes if p.role == "background")
    return _channel(
        "mt", mt_cuts(switches.vsjet_wp, switches.vsele_wp), processes, channel_samples, working_point_regions(),
        categories=categories(switches.vsjet_wp, switches.vsele_wp),
        variations=tuple(es_variation(shift) for shift in ES_GRID) + (mc_variations(mc_weights(switches.vsjet_wp, switches.vsele_wp)["tau_id"]) if switches.shape_systematics else ()),
        estimators=(DataMinus("QCD", "same_sign", subtract, clip_negative=True), TemplateShift(f"CMS_emb_ttbar_contamination_{ERA_TAG}", SIGNAL, "TTT", 0.1)),
    )


def mm_channel(channel_samples: tuple[Sample, ...]) -> Channel:
    processes = mm_processes()
    return _channel(
        "mm", mm_cuts(), processes, channel_samples,
        categories=(Category(CONTROL_CATEGORY, "(m_vis > 70) & (m_vis < 110)", Variable("m_vis", "m_vis", CONTROL_REGION_EDGES)),),
        estimators=(DataMinus("QCD", "same_sign", (MUON_EMBEDDED, "W"), clip_negative=True),),
    )


def build(config: RunConfig) -> Analysis:
    switches = parse(TauIdSwitches, config.switches)
    by_channel = samples(sample_database(config), switches.sample_lists, ["mt", "mm"])
    return Analysis(
        name="tau_id_es_embedding_2018",
        era=ERA,
        lumi_pb=LUMI_PB,
        signal=None,  # the fitted EMB is a background here; MorphingTauID2017 makes it the signal of its datacards
        channels={"mt": mt_channel(switches, by_channel["mt"]), "mm": mm_channel(by_channel["mm"])},
        measurement=TauIdEsMeasurement(switches.vsjet_wp, switches.vsele_wp, ES_GRID),
    )
