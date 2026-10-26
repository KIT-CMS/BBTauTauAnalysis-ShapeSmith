"""The SM fake-factor measurement of one era (`shapesmith measure`): QCD and ttbar fake factors of every hadronic tau,
their fractions, the QCD DR->SR correction and the non-closures, in the method of jvoss's TauFakeFactors
configuration FF_Updated configs/non_res_HH/2018 (tables in ff_tables.py). Two payload variants (v7 spec E18, 6.8;
switch ff_processes): QCD + ttbar (output fake_factors/<era>/) and QCD + W + ttbar (fake_factors_w/<era>/).

W+jets (et and mt, the variant QCD + W + ttbar): fake factors from data in the DR mt_1 > 70, no b tag (>= 2 jets),
isolated lepton, opposite sign, minus every MC but W and minus QCD from the same-sign version of each region (data
minus every MC, TauFakeFactors' QCD_SS_estimate); the non-closures in the DR; the DR->SR correction from W MC (the
orthogonal fake factors from W MC in the DR, the correction in mt_1 from the baseline, i.e. from no to at least one b
tag and from high to low mT, the main uncertainty); three-process fractions QCD (data minus every MC), W+jets and
ttbar (both MC). tt has no W+jets fake factors.

Eras: every era, and the combined era 2016 (constants.COMBINED_ERAS): one measurement of both 2016 halves, each sample
normalised to the luminosity of its half, one payload fake_factors(_w)/2016/; a run per half (2016preVFP,
2016postVFP) with the same tables is the cross-check. The tables of an era come from ff_tables.tables(era): outside
2018 (and for W+jets in every era) their edges are placeholders until `measure --suggest-binning` gives them; until
then `measure` refuses.

Regions: named cut replacements on the measurement baseline, which is the analysis baseline (cuts.baseline_cuts) with
its lepton vetoes combined into one cut, so that the ttbar scale regions invert them at once. The tau ID is
cuts.isolated/anti_isolated, as in the analysis. Per leg (tt: the leading "" and the subleading "_subleading" tau):
- QCD: same sign, >= 2 jets instead of the b tag, lt mt_1 < 50; SR-like isolated, AR-like the leg anti-isolated;
- QCD orthogonal (DR->SR fake factors): the same with, in lt, the lepton isolation !(0.05 <= iso_1 <= 0.15) and
  iso_1 < 0.4 (et 0.02; user decision U5), in tt the other tau anti-isolated;
- QCD DR->SR (the correction itself): the orthogonal regions with opposite sign (lt: mt_1 < 70 of the baseline);
- ttbar: SR the baseline, AR the leg anti-isolated; scale regions the same with inverted lepton vetoes, and their
  same-sign versions for the QCD there;
- fractions: the AR (lt); in tt both taus anti-isolated for both legs (user decision U6, tt_fraction_region:
  both_anti_iso) or, per leg, its AR (tt_fraction_region: application, the v6 choice).
The skim is the analysis control-region skim (both charges, VVVLoose taus, >= 2 jets or the b tag, lt iso_1 < 0.5)
without the lepton vetoes. The selections, triggers and weights are those of the analysis in the run's era (v7 spec
6.2: the trigger of the era, lt |eta_1| < 2.1, the 2018 HEM veto, the prefiring weight); the binning tables are the
2018 ones (ff_tables.py).

Processes: data; the genuine taus from EMB (in the embedded channels) or the MC T parts; the L and J parts of
DY/TT/ST/VV; W. The fake factors, fractions and non-closures subtract all of them (the ttbar data/MC factor all but
TTJ); the three DR->SR steps subtract the MC T parts instead of EMB, as TauFakeFactors (DR_SR use_embedding: false).

Differences to jvoss's configuration:
- et ttbar: jvoss cuts mt_1 < 70 only in the AR (inflating his MC fake factor about 1.9x); here in SR and AR as the
  analysis, and in the ttbar scale regions (mt ttbar: jvoss has no mt_1 cut, < 0.2 % effect).
- jets: n_jets >= 2 only; jpt >= 20 is implied, jvoss's jj_deltaR > 0.4 (tt >= 0.4) is dropped (<= 1e-4 of events).
- tt: jvoss's dilepton_veto in the QCD regions is dropped (not produced for tt).
- b tag: (n_bjets >= 1) & (bpair_pt_2 > 0) for jvoss's n_bjets >= 1 && bpair_pt_{1,2} >= 20 (the same events), and
  iso_1 < 0.15 for 0 <= iso_1 <= 0.15 (the same events at float precision).
- triggers and offline thresholds: the analysis ones (et: ele32 with pt_1 > 33, jvoss ele32 || ele35).
- tt fraction AR (both_anti_iso): the other tau anti-isolated for jvoss's fails Medium (the same events with VVVLoose
  taus).
- weights: the analysis weights, with the b-tag weight (jvoss: none), without jvoss's Z pT reweighting, the vsJet
  SF on genuine taus in every region (jvoss: the Medium SF only for a tau-ID cut in the region, so failing genuine
  taus of the tt fraction AR get none), and the analysis embedding weights.
- samples: the analysis sample list, e.g. the inclusive DY sample for npartons == 0 (jvoss: the pT(Z) bins only);
  like jvoss, TTV, EWK, single Higgs and the signal are not subtracted.
- the et/mt fraction split has the n_jets edges [-0.5, 2.5, 22.5] of its categories (user decision U4).
- the SystMCShiftDown of the corrections is smoothed at the centres of mass, as every other curve (user decision
  U13); TauFakeFactors smooths it at the bin centres, because a deepcopy drops the centres of mass.
"""
from __future__ import annotations

from shapesmith.config import RunConfig
from shapesmith.measurements.fake_factors import DataScale, DrSr, FakeFactorMeasurement, Fractions, Leg, ProcessFF, SameSignQCD
from shapesmith.model import Analysis, AnalysisError, Channel, Process, Region, Sample, Selection

from bbtautau_shapesmith import cuts, ff_tables
from bbtautau_shapesmith.constants import LT_CHANNELS, TAU_CHANNELS, TAU_LEGS, check_measurement_era, lumi_pb
from bbtautau_shapesmith.processes import EMBEDDED, SPLITS, PLOT_GROUP
from bbtautau_shapesmith.samples import sample_database, samples
from bbtautau_shapesmith.switches import FF_VARIANTS, FakeFactorSwitches, parse
from bbtautau_shapesmith.weights import embedding_weights, group_weights, mc_weights

GROUPS = ("DY", "TT", "ST", "VV")  # split into T, L and J
VETO_CUTS = ("extraelec_veto", "extramuon_veto", "dilepton_veto")
JETS = "(n_jets >= 2)"
W_DR = {"mt_cut": "(mt_1 > 70)", "b_tagging": "(n_bjets == 0) & (n_jets >= 2)"}  # the W+jets determination region
W_MC = "W"  # the W+jets MC process
LEPTON_SIDEBAND = {  # jvoss's DR->SR lepton isolation (U5): the complement of his orthogonal window, below his n-tuples' 0.4
    "mt": "(~((iso_1 >= 0.05) & (iso_1 <= 0.15))) & (iso_1 < 0.4)",
    "et": "(~((iso_1 >= 0.02) & (iso_1 <= 0.15))) & (iso_1 < 0.4)",
}


def parts(part: str) -> tuple[str, ...]:
    return tuple(SPLITS[group][part] for group in GROUPS)


def measurement_processes(channel: str, era: str, embedding: bool, dy_correction: str = "none") -> tuple[Process, ...]:
    mc, genmatch = mc_weights(channel, era), cuts.genmatch_cuts(channel)

    def simulated(name: str, group: str, part: str | None = None, role: str = "background") -> Process:
        weights = {**mc, **group_weights(group, era, dy_correction)}
        return Process(name, group, role, PLOT_GROUP[group], Selection(cuts={"genmatch": genmatch[part]} if part else {}, weights=weights))

    result = [Process("data", "data", "data", "data")]
    if embedding:
        result.append(Process(EMBEDDED, EMBEDDED, "background", EMBEDDED, Selection(cuts={"genmatch": genmatch["T"]}, weights=embedding_weights(channel, era))))
    for part in ("T", "L", "J"):
        role = "auxiliary" if part == "T" and embedding else "background"  # with embedding, T is subtracted only in the DR->SR steps
        result += [simulated(SPLITS[group][part], group, part, role) for group in GROUPS]
    result.append(simulated("W", "W"))
    return tuple(result)


def baseline(channel: str, era: str) -> dict[str, str]:
    """The analysis baseline with the lepton vetoes as one cut."""
    base = cuts.baseline_cuts(channel, era)
    vetoes = [base.pop(name) for name in VETO_CUTS if name in base]
    return {"lepton_vetoes": " & ".join(vetoes), **base}


def skim(channel: str, era: str) -> dict[str, str]:
    return {name: expr for name, expr in cuts.skim_cuts(channel, era, control_regions=True).items() if name not in VETO_CUTS}


def tau_id(channel: str, anti_isolated: tuple[int, ...]) -> str:
    return " & ".join(cuts.anti_isolated(leg) if leg in anti_isolated else cuts.isolated(leg) for leg in TAU_LEGS[channel])


def leg_regions(channel: str, era: str, leg: int, suffix: str, tt_fraction_region: str = "both_anti_iso", wjets: bool = False) -> dict[str, dict[str, str]]:
    """The cut replacements of the regions of one leg, by region name (with `wjets` in et and mt also those of the
    W+jets fake factors)."""
    lt = channel in LT_CHANNELS
    other = tuple(t for t in TAU_LEGS[channel] if t != leg)
    fraction_legs = (leg,) if lt or tt_fraction_region == "application" else TAU_LEGS[channel]
    qcd = {"os": cuts.SAME_SIGN, "b_tagging": JETS, **({"mt_cut": "(mt_1 < 50)"} if lt else {})}
    # the DR->SR regions are orthogonal by the lepton isolation (lt) or by the other tau anti-isolated (tt)
    sideband, anti_other = ({"lepton_iso": LEPTON_SIDEBAND[channel]}, ()) if lt else ({}, other)
    inverted = {"lepton_vetoes": f"~({baseline(channel, era)['lepton_vetoes']})"}
    regions = {
        f"QCD{suffix}_sr_like": {**qcd, "tau_iso": tau_id(channel, ())},
        f"QCD{suffix}_ar_like": {**qcd, "tau_iso": tau_id(channel, (leg,))},
        f"QCD{suffix}_orthogonal_sr_like": {**qcd, **sideband, "tau_iso": tau_id(channel, anti_other)},
        f"QCD{suffix}_orthogonal_ar_like": {**qcd, **sideband, "tau_iso": tau_id(channel, (leg, *anti_other))},
        f"QCD{suffix}_dr_sr_sr_like": {"b_tagging": JETS, **sideband, "tau_iso": tau_id(channel, anti_other)},
        f"QCD{suffix}_dr_sr_ar_like": {"b_tagging": JETS, **sideband, "tau_iso": tau_id(channel, (leg, *anti_other))},
        f"ttbar{suffix}_sr": {},
        f"ttbar{suffix}_ar": {"tau_iso": tau_id(channel, (leg,))},
        f"process_fractions{suffix}_ar": {"tau_iso": tau_id(channel, fraction_legs)},
    }
    for sign, charge in (("", {}), ("_ss", {"os": cuts.SAME_SIGN})):
        regions[f"ttbar{suffix}_scale_sr_like{sign}"] = {**inverted, **charge}
        regions[f"ttbar{suffix}_scale_ar_like{sign}"] = {**inverted, **charge, "tau_iso": tau_id(channel, (leg,))}
    if wjets and lt:
        for sign, charge in (("", {}), ("_ss", {"os": cuts.SAME_SIGN})):
            regions[f"Wjets_sr_like{sign}"] = {**W_DR, **charge}
            regions[f"Wjets_ar_like{sign}"] = {**W_DR, **charge, "tau_iso": tau_id(channel, (leg,))}
        regions["Wjets_sr"] = {}  # the DR->SR correction from W MC: the baseline
        regions["Wjets_ar"] = {"tau_iso": tau_id(channel, (leg,))}
    return regions


def wjets_process(table: dict, genuine: tuple[str, ...], fakes: tuple[str, ...]) -> ProcessFF:
    """The W+jets fake factors of an lt leg: from data minus every MC but W and minus the same-sign QCD in the DR, the
    DR->SR correction from W MC."""
    every_mc = genuine + fakes
    qcd = SameSignQCD({"Wjets_sr_like": "Wjets_sr_like_ss", "Wjets_ar_like": "Wjets_ar_like_ss"}, every_mc)
    dr_sr = DrSr("Wjets_sr_like", "Wjets_ar_like", "Wjets_sr", "Wjets_ar", (), table["dr_sr"], table["dr_sr_non_closures"], target=W_MC)
    return ProcessFF("Wjets", "data", tuple(p for p in every_mc if p != W_MC), "Wjets_sr_like", "Wjets_ar_like", table["split"], table["fake_factors"],
                     table["non_closures"], dr_sr, qcd=qcd)


def legs(channel: str, embedding: bool, tables: dict | None = None, wjets: bool = False) -> tuple[Leg, ...]:
    """The measurement of every hadronic tau, with the region names of leg_regions; `tables` of the era (default the
    2018 ones), with `wjets` the W+jets fake factors in et and mt."""
    genuine = (EMBEDDED,) if embedding else parts("T")
    fakes = parts("L") + parts("J") + (W_MC,)
    tables = (tables or ff_tables.tables("2018", wjets)[0])[channel]
    with_w = wjets and channel in LT_CHANNELS
    result = []
    for leg, suffix in zip(TAU_LEGS[channel], ("", "_subleading")):
        qcd_table, ttbar_table = tables[f"QCD{suffix}"], tables[f"ttbar{suffix}"]
        dr_sr = DrSr(
            f"QCD{suffix}_orthogonal_sr_like", f"QCD{suffix}_orthogonal_ar_like", f"QCD{suffix}_dr_sr_sr_like", f"QCD{suffix}_dr_sr_ar_like",
            parts("T") + fakes, qcd_table["dr_sr"], qcd_table["dr_sr_non_closures"],
        )
        qcd = ProcessFF(
            "QCD", "data", genuine + fakes, f"QCD{suffix}_sr_like", f"QCD{suffix}_ar_like", qcd_table["split"], qcd_table["fake_factors"],
            qcd_table["non_closures"], dr_sr,
        )
        scale = DataScale(
            f"ttbar{suffix}_scale_sr_like", f"ttbar{suffix}_scale_ar_like", f"ttbar{suffix}_scale_sr_like_ss", f"ttbar{suffix}_scale_ar_like_ss",
            genuine + fakes,
        )
        ttbar = ProcessFF(
            "ttbar", "TTJ", (), f"ttbar{suffix}_sr", f"ttbar{suffix}_ar", ttbar_table["split"], ttbar_table["fake_factors"], ttbar_table["non_closures"],
            scale=scale,
        )
        fractions_table = tables[f"process_fractions{suffix}"]
        fractions = Fractions(f"process_fractions{suffix}_ar", genuine + fakes, "TTJ", fractions_table["split"], fractions_table["binned"], wjets=W_MC if with_w else None)
        result.append(Leg(suffix, qcd, ttbar, fractions, wjets_process(tables["Wjets"], genuine, fakes) if with_w else None))
    return tuple(result)


def channel(name: str, era: str, switches: FakeFactorSwitches, channel_samples: tuple[Sample, ...], measurement: FakeFactorMeasurement) -> Channel:
    table = measurement_processes(name, era, switches.embedded(name), switches.dy_correction)
    groups = {p.group for p in table}
    regions = {}
    for leg, suffix in zip(TAU_LEGS[name], ("", "_subleading")):
        regions.update(leg_regions(name, era, leg, suffix, switches.tt_fraction_region, switches.wjets))
    return Channel(
        name=name,
        samples=tuple(s for s in channel_samples if s.group in groups),
        skim=skim(name, era),
        cuts=baseline(name, era),
        processes=table,
        regions=tuple(Region(region, replace_cuts=replaced) for region, replaced in regions.items()),
        keep_columns=tuple(sorted(measurement.columns(name))),
    )


def placeholder_note(era: str, placeholders: list[str], channels: list[str]) -> str | None:
    """Why the edges of the run's channels are not yet the era's own (FakeFactorMeasurement.placeholder_edges)."""
    missing = [name for name in placeholders if name.split(" ", 1)[0] in channels]
    if not missing:
        return None
    return f"{len(missing)} quantities have no edges of {ff_tables.TABLE_ERA[era]} in ff_tables.PER_ERA_EDGES, e.g. {', '.join(missing[:3])}"


def build(config: RunConfig) -> Analysis:
    era = check_measurement_era(config.era)
    switches = parse(FakeFactorSwitches, config.switches)
    if not config.channels or set(config.channels) - set(TAU_CHANNELS):
        raise AnalysisError(f"the fake-factor measurement supports the channels {', '.join(TAU_CHANNELS)}")
    switches.check_embedding(config.channels)
    tables, placeholders = ff_tables.tables(era, switches.wjets)
    measurement = FakeFactorMeasurement(
        {name: legs(name, switches.embedded(name), tables, switches.wjets) for name in config.channels},
        name=FF_VARIANTS[tuple(switches.ff_processes)],
        placeholder_edges=placeholder_note(era, placeholders, config.channels),
    )
    by_channel = samples(sample_database(config), switches.sample_lists, config.channels, era)
    return Analysis(
        name=f"hh_bbtautau_{era}_v15_fake_factors",
        era=era,
        lumi_pb=lumi_pb(era),
        signal=None,
        channels={name: channel(name, era, switches, by_channel[name], measurement) for name in config.channels},
        measurement=measurement,
        # no event_key: the ff_measurement skims hold no run and lumi (keep_columns), so the repeated events of the 2018D
        # mutau embedding stay in (252 of its 147 416 mt rows repeat an event number); EVENT_KEY here needs a new skim
    )
