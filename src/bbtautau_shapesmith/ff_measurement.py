"""The SM 2018 fake-factor measurement (`shapesmith measure`): QCD and ttbar fake factors of every hadronic tau,
their fractions, the QCD DR->SR correction and the non-closures, in the method of jvoss's TauFakeFactors
configuration FF_Updated configs/non_res_HH/2018 (tables in ff_tables.py).

Regions: named cut replacements on the measurement baseline, which is the analysis baseline (cuts.baseline_cuts) with
its lepton vetoes combined into one cut, so that the ttbar scale regions invert them at once. The tau ID is
cuts.isolated/anti_isolated, as in the analysis. Per leg (tt: the leading "" and the subleading "_subleading" tau):
- QCD: same sign, >= 2 jets instead of the b tag, lt mt_1 < 50; SR-like isolated, AR-like the leg anti-isolated;
- QCD orthogonal (DR->SR fake factors): the same with, in lt, the lepton isolation !(0.05 <= iso_1 <= 0.15) and
  iso_1 < 0.4 (et 0.02; user decision U5), in tt the other tau anti-isolated;
- QCD DR->SR (the correction itself): the orthogonal regions with opposite sign (lt: mt_1 < 70 of the baseline);
- ttbar: SR the baseline, AR the leg anti-isolated; scale regions the same with inverted lepton vetoes, and their
  same-sign versions for the QCD there;
- fractions: the AR (lt), in tt both taus anti-isolated for both legs (user decision U6).
The skim is the analysis control-region skim (both charges, VVVLoose taus, >= 2 jets or the b tag, lt iso_1 < 0.5)
without the lepton vetoes.

Processes: data; the genuine taus from EMB (embedding) or the MC T parts; the L and J parts of DY/TT/ST/VV; W. The
fake factors, fractions and non-closures subtract all of them (the ttbar data/MC factor all but TTJ); the three DR->SR steps
subtract the MC T parts instead of EMB, as TauFakeFactors (DR_SR use_embedding: false).

Differences to jvoss's configuration:
- et ttbar: jvoss cuts mt_1 < 70 only in the AR (inflating his MC fake factor about 1.9x); here in SR and AR as the
  analysis, and in the ttbar scale regions (mt ttbar: jvoss has no mt_1 cut, < 0.2 % effect).
- jets: n_jets >= 2 only; jpt >= 20 is implied, jvoss's jj_deltaR > 0.4 (tt >= 0.4) is dropped (<= 1e-4 of events).
- tt: jvoss's dilepton_veto in the QCD regions is dropped (not produced for tt).
- b tag: (n_bjets >= 1) & (bpair_pt_2 > 0) for jvoss's n_bjets >= 1 && bpair_pt_{1,2} >= 20 (the same events), and
  iso_1 < 0.15 for 0 <= iso_1 <= 0.15 (the same events at float precision).
- triggers and offline thresholds: the analysis ones (et: ele32 with pt_1 > 33, jvoss ele32 || ele35).
- tt fraction AR: the other tau anti-isolated for jvoss's fails Medium (the same events with VVVLoose taus).
- weights: the analysis weights, with the b-tag weight (jvoss: none), without jvoss's Z pT reweighting, the vsJet
  SF on genuine taus in every region (jvoss: the Medium SF only for a tau-ID cut in the region, so failing genuine
  taus of the tt fraction AR get none), and the analysis embedding weights.
- samples: the analysis sample list, e.g. the inclusive DY sample for npartons == 0 (jvoss: the pT(Z) bins only);
  like jvoss, TTV, EWK, single Higgs and the signal are not subtracted.
- the et/mt fraction split has the n_jets edges [-0.5, 2.5, 22.5] of its categories (user decision U4).
"""
from __future__ import annotations

from shapesmith.config import RunConfig
from shapesmith.measurements.fake_factors import DataScale, DrSr, FakeFactorMeasurement, Fractions, Leg, ProcessFF
from shapesmith.model import Analysis, AnalysisError, Channel, Process, Region, Sample, Selection

from bbtautau_shapesmith import cuts
from bbtautau_shapesmith.constants import ERA, LT_CHANNELS, LUMI_PB, TAU_CHANNELS, TAU_LEGS
from bbtautau_shapesmith.ff_tables import TABLES
from bbtautau_shapesmith.processes import EMBEDDED, SPLITS, PLOT_GROUP
from bbtautau_shapesmith.samples import sample_database, samples
from bbtautau_shapesmith.switches import FakeFactorSwitches, parse
from bbtautau_shapesmith.weights import TOP_PT, embedding_weights, mc_weights

GROUPS = ("DY", "TT", "ST", "VV")  # split into T, L and J
VETO_CUTS = ("extraelec_veto", "extramuon_veto", "dilepton_veto")
JETS = "(n_jets >= 2)"
LEPTON_SIDEBAND = {  # jvoss's DR->SR lepton isolation (U5): the complement of his orthogonal window, below his n-tuples' 0.4
    "mt": "(~((iso_1 >= 0.05) & (iso_1 <= 0.15))) & (iso_1 < 0.4)",
    "et": "(~((iso_1 >= 0.02) & (iso_1 <= 0.15))) & (iso_1 < 0.4)",
}


def parts(part: str) -> tuple[str, ...]:
    return tuple(SPLITS[group][part] for group in GROUPS)


def measurement_processes(channel: str, embedding: bool) -> tuple[Process, ...]:
    mc, genmatch = mc_weights(channel), cuts.genmatch_cuts(channel)

    def simulated(name: str, group: str, part: str | None = None, role: str = "background") -> Process:
        weights = {**mc, **TOP_PT} if group == "TT" else mc
        return Process(name, group, role, PLOT_GROUP[group], Selection(cuts={"genmatch": genmatch[part]} if part else {}, weights=weights))

    result = [Process("data", "data", "data", "data")]
    if embedding:
        result.append(Process(EMBEDDED, EMBEDDED, "background", EMBEDDED, Selection(cuts={"genmatch": genmatch["T"]}, weights=embedding_weights(channel))))
    for part in ("T", "L", "J"):
        role = "auxiliary" if part == "T" and embedding else "background"  # with embedding, T is subtracted only in the DR->SR steps
        result += [simulated(SPLITS[group][part], group, part, role) for group in GROUPS]
    result.append(simulated("W", "W"))
    return tuple(result)


def baseline(channel: str) -> dict[str, str]:
    """The analysis baseline with the lepton vetoes as one cut."""
    base = cuts.baseline_cuts(channel)
    vetoes = [base.pop(name) for name in VETO_CUTS if name in base]
    return {"lepton_vetoes": " & ".join(vetoes), **base}


def skim(channel: str) -> dict[str, str]:
    return {name: expr for name, expr in cuts.skim_cuts(channel, control_regions=True).items() if name not in VETO_CUTS}


def tau_id(channel: str, anti_isolated: tuple[int, ...]) -> str:
    return " & ".join(cuts.anti_isolated(leg) if leg in anti_isolated else cuts.isolated(leg) for leg in TAU_LEGS[channel])


def leg_regions(channel: str, leg: int, suffix: str) -> dict[str, dict[str, str]]:
    """The cut replacements of the regions of one leg, by region name."""
    lt = channel in LT_CHANNELS
    other = tuple(t for t in TAU_LEGS[channel] if t != leg)
    qcd = {"os": cuts.SAME_SIGN, "b_tagging": JETS, **({"mt_cut": "(mt_1 < 50)"} if lt else {})}
    # the DR->SR regions are orthogonal by the lepton isolation (lt) or by the other tau anti-isolated (tt)
    sideband, anti_other = ({"lepton_iso": LEPTON_SIDEBAND[channel]}, ()) if lt else ({}, other)
    inverted = {"lepton_vetoes": f"~({baseline(channel)['lepton_vetoes']})"}
    regions = {
        f"QCD{suffix}_sr_like": {**qcd, "tau_iso": tau_id(channel, ())},
        f"QCD{suffix}_ar_like": {**qcd, "tau_iso": tau_id(channel, (leg,))},
        f"QCD{suffix}_orthogonal_sr_like": {**qcd, **sideband, "tau_iso": tau_id(channel, anti_other)},
        f"QCD{suffix}_orthogonal_ar_like": {**qcd, **sideband, "tau_iso": tau_id(channel, (leg, *anti_other))},
        f"QCD{suffix}_dr_sr_sr_like": {"b_tagging": JETS, **sideband, "tau_iso": tau_id(channel, anti_other)},
        f"QCD{suffix}_dr_sr_ar_like": {"b_tagging": JETS, **sideband, "tau_iso": tau_id(channel, (leg, *anti_other))},
        f"ttbar{suffix}_sr": {},
        f"ttbar{suffix}_ar": {"tau_iso": tau_id(channel, (leg,))},
        f"process_fractions{suffix}_ar": {"tau_iso": tau_id(channel, (leg,) if lt else TAU_LEGS[channel])},
    }
    for sign, charge in (("", {}), ("_ss", {"os": cuts.SAME_SIGN})):
        regions[f"ttbar{suffix}_scale_sr_like{sign}"] = {**inverted, **charge}
        regions[f"ttbar{suffix}_scale_ar_like{sign}"] = {**inverted, **charge, "tau_iso": tau_id(channel, (leg,))}
    return regions


def legs(channel: str, embedding: bool) -> tuple[Leg, ...]:
    """The measurement of every hadronic tau, with the region names of leg_regions."""
    genuine = (EMBEDDED,) if embedding else parts("T")
    fakes = parts("L") + parts("J") + ("W",)
    tables = TABLES[channel]
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
        fractions = Fractions(f"process_fractions{suffix}_ar", genuine + fakes, "TTJ", fractions_table["split"], fractions_table["binned"])
        result.append(Leg(suffix, qcd, ttbar, fractions))
    return tuple(result)


def channel(name: str, embedding: bool, channel_samples: tuple[Sample, ...], measurement: FakeFactorMeasurement) -> Channel:
    table = measurement_processes(name, embedding)
    groups = {p.group for p in table}
    regions = {}
    for leg, suffix in zip(TAU_LEGS[name], ("", "_subleading")):
        regions.update(leg_regions(name, leg, suffix))
    return Channel(
        name=name,
        samples=tuple(s for s in channel_samples if s.group in groups),
        skim=skim(name),
        cuts=baseline(name),
        processes=table,
        regions=tuple(Region(region, replace_cuts=replaced) for region, replaced in regions.items()),
        keep_columns=tuple(sorted(measurement.columns(name))),
    )


def build(config: RunConfig) -> Analysis:
    switches = parse(FakeFactorSwitches, config.switches)
    if config.era != ERA or not config.channels or set(config.channels) - set(TAU_CHANNELS):
        raise AnalysisError(f"the fake-factor measurement supports era {ERA} and the channels {', '.join(TAU_CHANNELS)}")
    measurement = FakeFactorMeasurement({name: legs(name, switches.embedding) for name in config.channels})
    by_channel = samples(sample_database(config), switches.sample_lists, config.channels)
    return Analysis(
        name="hh_bbtautau_2018_v15_fake_factors",
        era=ERA,
        lumi_pb=LUMI_PB,
        signal=None,
        channels={name: channel(name, switches.embedding, by_channel[name], measurement) for name in config.channels},
        measurement=measurement,
    )
