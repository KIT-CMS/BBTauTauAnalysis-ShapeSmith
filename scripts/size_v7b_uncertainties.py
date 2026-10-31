"""Size the uncertainties of the v7b fake-factor runs (study_uncertainties.py) from the skims of a control run.

Reads the nominal-region events of every simulated process and of the embedded sample of one era from the skims of the
v7b control run (configs/smrun2_binned_v7b.yaml: embedding in et and mt, MC genuine taus in tt, jet fakes from MC) and
evaluates the correctionlib payloads per event:
- embedding (et, mt): the vsJet SF and the energy scale of the CROWN embedding payload (up/down of the 1 sigma interval),
  per decay-mode category 0, 1, 10+11; the energy scale through the tau pT threshold of the production and mT < 70 GeV
  (the MET recoils against the tau as in the energy-scale grid of the tau-ID measurement);
- simulation (et, mt, tt, every leg with gen_match 5): the POG tau energy scale per decay mode through the same
  acceptance (tt: both taus above 40 GeV), the POG vsJet SF components (syst_alleras, syst_<era>, stat1/2 per decay mode,
  syst_TES for information), the POG di-tau trigger SF in tt per decay mode;
- simulation: the statistical part of the muon_Z ID and isolation SFs (mt, em) and the EGM electron reconstruction SF
  (et, em; em from the dilepton run);
- data: the decay-mode composition of the anti-isolated region (the application region of the fake factors).
Every number is a ratio of weighted yields (varied / nominal) of the nominal selection. Writes <out>/<era>.json.

    python scripts/size_v7b_uncertainties.py --era 2018 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import correctionlib
import numpy as np
from shapesmith.config import load_analysis, load_config
from shapesmith.events import event_weights, lumi, select
from shapesmith.expressions import columns_of, mask
from shapesmith.model import Region
from shapesmith.store import SKIM_COLUMNS, read_skims

from bbtautau_shapesmith.constants import TAU_VS_ELE_WP

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
CROWN = Path("/work/sdaigler/bbtautau/KingMaker/CROWN/analysis_configurations/bbtautau")
EMBEDDING_PAYLOAD = CROWN / "payloads/tau_embedding/Run2-{era}-UL-NanoAODv15/DeepTau2018v2p5_id_es_embedding{era}UL.json.gz"
POG_TAU = "/cvmfs/cms-griddata.cern.ch/cat/metadata/TAU/Run2-{era}-UL-NanoAODv15/2025-11-27/tau.json.gz"
POG_MUON = "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run2-{era}-UL-NanoAODv9/2024-07-02/muon_Z.json.gz"
POG_ELECTRON = "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run2-{era}-UL-NanoAODv15/2025-12-05/electron.json.gz"
POG_ERA_TOKEN = {"2016preVFP": "2016_preVFP", "2016postVFP": "2016_postVFP", "2017": "2017", "2018": "2018"}
EMB_TOKENS = {"0": (0,), "1": (1,), "1011": (10, 11)}
MC_DECAY_MODES = (0, 1, 10, 11)
COLUMNS = {"pt_1", "pt_2", "eta_1", "eta_2", "phi_1", "phi_2", "mass_1", "mass_2", "met", "metphi", "mt_1",
           "tau_decaymode_1", "tau_decaymode_2", "gen_match_1", "gen_match_2"}
TT_PT = 40.0

logger = logging.getLogger("size")


def rows(config, analysis, channel_name, process_name, region="nominal", drop=()):
    """The events and weights of a process in a region, with the named cuts of the channel dropped."""
    channel = analysis.channel(channel_name)
    process = channel.process(process_name)
    base = channel.region(region)
    region = Region(base.name, {**base.replace_cuts, **{name: "True" for name in drop}}, base.add_weights, base.replace_weights)
    cuts, weights = select(channel, process, region)
    if drop:  # the skim cuts repeat the channel cuts; leave the dropped ones out there too
        skim = {expr for name, expr in channel.skim.items() if name in drop}
        cuts = [c for c in cuts if c not in skim]
    nicks = [s.nick for s in channel.samples_of(process.group)]
    if not nicks:
        return None, None
    frame = read_skims(config.skim_dir, channel.name, nicks, set(SKIM_COLUMNS) | columns_of(cuts + weights) | (COLUMNS & _columns(config, channel, nicks)))
    frame = frame[mask(frame, cuts)].reset_index(drop=True)
    return frame, event_weights(frame, weights, lumi(analysis, channel, process))


def _columns(config, channel, nicks):
    from shapesmith.store import schema
    return schema(config.skim_dir, channel.name, nicks)


def ratio(w, factor):
    total = w.sum()
    return float((w * factor).sum() / total) if total > 0 else 1.0


def shifted_mt(frame, s):
    """mT of the light lepton with the tau scaled by s and the MET recoiling (as tau_id_measurement.es_variation)."""
    recoil = 1.0 - s
    mx = frame.met * np.cos(frame.metphi) + recoil * frame.pt_2 * np.cos(frame.phi_2)
    my = frame.met * np.sin(frame.metphi) + recoil * frame.pt_2 * np.sin(frame.phi_2)
    met, phi = np.hypot(mx, my), np.arctan2(my, mx)
    return np.sqrt(2 * frame.pt_1 * met * (1 - np.cos(frame.phi_1 - phi)))


def lt_acceptance(nominal, wide, threshold, scale_nominal, scale_wide):
    """Yield ratio of the nominal selection of et/mt when the taus are scaled per event: the tau pT threshold of the
    production (events below it are not stored: an upward shift gains the mirrored count above it) and mT < 70 GeV
    (`wide` has no mT cut, so events migrate in and out)."""
    frame, w = nominal
    wf, ww = wide
    pt = frame.pt_2 * scale_nominal
    lost_pt = (pt < threshold)
    # gained below the threshold, mirrored: the events in [thr, thr * s) for s > 1
    gain_pt = (frame.pt_2 < threshold * scale_nominal) & (scale_nominal > 1)
    keep = ~lost_pt & (shifted_mt(frame, scale_nominal) < 70)
    enter = (wf.mt_1 >= 70) & (shifted_mt(wf, scale_wide) < 70) & (wf.pt_2 * scale_wide >= threshold)
    total = w.sum()
    return float((w[keep].sum() + w[gain_pt].sum() + ww[enter].sum()) / total)


def tt_acceptance(frame, w, scale_1, scale_2):
    """Yield ratio of tt with both taus scaled per leg: both above 40 GeV, an upward shift gaining the mirrored count."""
    pass_now = (frame.pt_1 * scale_1 >= TT_PT) & (frame.pt_2 * scale_2 >= TT_PT)
    gain = ((frame.pt_1 < TT_PT * scale_1) & (scale_1 > 1)) | ((frame.pt_2 < TT_PT * scale_2) & (scale_2 > 1))
    return float((w[pass_now].sum() + w[gain].sum()) / w.sum())


def vectorised(correction, *args):
    return np.asarray(correction.evaluate(*args), dtype=float)


def embedding_numbers(config, analysis, era, channel):
    frame, w = rows(config, analysis, channel, "EMB")
    wide = rows(config, analysis, channel, "EMB", drop=("mt_cut",))
    cset = correctionlib.CorrectionSet.from_file(str(EMBEDDING_PAYLOAD).format(era=era))
    vsjet, es = cset["DeepTau2018v2p5VSjet"], cset["tau_energy_scale_dm_binned"]
    wp_vse = TAU_VS_ELE_WP[channel]
    threshold = float(np.floor(frame.pt_2.min()))

    def sf(f, syst):
        return vectorised(vsjet, f.pt_2.to_numpy(), f.tau_decaymode_2.to_numpy().astype(int), np.full(len(f), 5), "Medium", wp_vse, syst, "dm")

    def scale(f, syst):
        return vectorised(es, f.pt_2.to_numpy(), f.eta_2.to_numpy(), f.tau_decaymode_2.to_numpy().astype(int), np.full(len(f), 5), "DeepTau2018v2p5", "Medium", wp_vse, syst)

    nominal_sf, nominal_es = sf(frame, "nom"), scale(frame, "nom")
    wide_es = scale(wide[0], "nom")
    result = {"threshold": threshold, "yield": float(w.sum()), "dm_fractions": {}, "id": {}, "es": {}}
    for token, dms in EMB_TOKENS.items():
        inside = frame.tau_decaymode_2.isin(dms).to_numpy()
        inside_wide = wide[0].tau_decaymode_2.isin(dms).to_numpy()
        result["dm_fractions"][token] = float(w[inside].sum() / w.sum())
        entry_id, entry_es = {}, {}
        for direction in ("up", "down"):
            factor = np.where(inside, sf(frame, direction) / nominal_sf, 1.0)
            entry_id[direction] = ratio(w, factor)
            s_nom = np.where(inside, scale(frame, direction) / nominal_es, 1.0)
            s_wide = np.where(inside_wide, scale(wide[0], direction) / wide_es, 1.0)
            entry_es[direction] = lt_acceptance((frame, w), wide, threshold, s_nom, s_wide)
            entry_es[f"{direction}_shift"] = float(np.average(s_nom[inside], weights=w[inside]) - 1) if inside.any() else 0.0
        result["id"][token], result["es"][token] = entry_id, entry_es
    return result


def mc_processes(analysis, channel):
    return [p.name for p in analysis.channel(channel).processes if p.role in ("signal", "background") and analysis.channel(channel).kind_of(p) == "mc"]


def mc_numbers(config, analysis, era, channel):
    """POG tau ES, vsJet components and (tt) di-tau trigger SF on every simulated process, per process."""
    cset = correctionlib.CorrectionSet.from_file(POG_TAU.format(era=era))
    tes, vsjet, trigger = cset["tau_energy_scale"], cset["DeepTau2018v2p5VSjet"], cset["tau_trigger"]
    token = POG_ERA_TOKEN[era]
    wp_vse = TAU_VS_ELE_WP[channel]
    legs = (1, 2) if channel == "tt" else (2,)
    components = ["syst_alleras", f"syst_{token}"] + [f"stat{i}_dm{dm}" for i in (1, 2) for dm in MC_DECAY_MODES] + [f"syst_TES_{token}_dm{dm}" for dm in MC_DECAY_MODES]
    result = {}
    for process in mc_processes(analysis, channel):
        frame, w = rows(config, analysis, channel, process)
        if frame is None or len(frame) == 0 or w.sum() <= 0:
            continue
        wide = rows(config, analysis, channel, process, drop=("mt_cut",)) if channel != "tt" else None
        entry = {"yield": float(w.sum()), "tes": {}, "vsjet": {}, "trigger": {}}
        genuine = {leg: (frame[f"gen_match_{leg}"] == 5).to_numpy() for leg in legs}
        dm = {leg: frame[f"tau_decaymode_{leg}"].to_numpy().astype(int) for leg in legs}
        dm_pog = {leg: np.where(dm[leg] == 2, 1, dm[leg]) for leg in legs}  # the POG treats DM 2 as DM 1 (K11)
        pt = {leg: frame[f"pt_{leg}"].to_numpy() for leg in legs}
        eta = {leg: frame[f"eta_{leg}"].to_numpy() for leg in legs}

        def es(leg, syst, f=frame):
            d = f[f"tau_decaymode_{leg}"].to_numpy().astype(int)
            g = f[f"gen_match_{leg}"].to_numpy().astype(int)
            return vectorised(tes, f[f"pt_{leg}"].to_numpy(), f[f"eta_{leg}"].to_numpy(), np.where(d == 2, 1, d), np.where(g == 5, 5, 0), "DeepTau2018v2p5", "Medium", wp_vse, syst)

        for decay_mode in MC_DECAY_MODES:
            entry["tes"][decay_mode] = {}
            for direction in ("up", "down"):
                scales = {}
                for leg in legs:
                    s = es(leg, direction) / es(leg, "nom")
                    scales[leg] = np.where(genuine[leg] & (dm_pog[leg] == decay_mode), s, 1.0)
                if channel == "tt":
                    entry["tes"][decay_mode][direction] = tt_acceptance(frame, w, scales[1], scales[2])
                else:
                    wf = wide[0]
                    dw = wf.tau_decaymode_2.to_numpy().astype(int)
                    sw = np.where((wf.gen_match_2 == 5).to_numpy() & (np.where(dw == 2, 1, dw) == decay_mode), es(2, direction, wf) / es(2, "nom", wf), 1.0)
                    threshold = float(np.floor(frame.pt_2.min()))
                    entry["tes"][decay_mode][direction] = lt_acceptance((frame, w), wide, threshold, scales[2], sw)
        nominal_sf = {leg: vectorised(vsjet, pt[leg], dm_pog[leg], np.full(len(frame), 5), "Medium", wp_vse, "nom", "dm") for leg in legs}
        for component in components:
            entry["vsjet"][component] = {}
            for direction in ("up", "down"):
                factor = np.ones(len(frame))
                for leg in legs:
                    varied = vectorised(vsjet, pt[leg], dm_pog[leg], np.full(len(frame), 5), "Medium", wp_vse, f"{component}_{direction}", "dm")
                    factor *= np.where(genuine[leg], varied / nominal_sf[leg], 1.0)
                entry["vsjet"][component][direction] = ratio(w, factor)
        if channel == "tt":
            nominal_trg = {leg: vectorised(trigger, pt[leg], dm_pog[leg], "ditau", "Medium", "sf", "nom") for leg in legs}
            for decay_mode in MC_DECAY_MODES:
                entry["trigger"][decay_mode] = {}
                for direction in ("up", "down"):
                    factor = np.ones(len(frame))
                    for leg in legs:
                        varied = vectorised(trigger, pt[leg], dm_pog[leg], "ditau", "Medium", "sf", direction)
                        factor *= np.where(dm_pog[leg] == decay_mode, varied / nominal_trg[leg], 1.0)
                    entry["trigger"][decay_mode][direction] = ratio(w, factor)
        entry["genuine_fraction"] = float(sum(w[genuine[leg]].sum() for leg in legs) / (len(legs) * w.sum()))
        entry["dm_fractions"] = {int(d): float(sum(w[genuine[leg] & (dm_pog[leg] == d)].sum() for leg in legs) / max(sum(w[genuine[leg]].sum() for leg in legs), 1e-12)) for d in MC_DECAY_MODES}
        result[process] = entry
        logger.info(f"{era} {channel} {process}: yield {w.sum():.2f}")
    return result


def lepton_numbers(config, analysis, era, channel, legs):
    """The muon_Z ID and isolation stat (relative, yield-weighted) and the EGM reco up/down of the light leptons."""
    muon = correctionlib.CorrectionSet.from_file(POG_MUON.format(era=era))
    electron = correctionlib.CorrectionSet.from_file(POG_ELECTRON.format(era=era))["UL-Electron-ID-SF"]
    result = {}
    for process in mc_processes(analysis, channel):
        frame, w = rows(config, analysis, channel, process)
        if frame is None or len(frame) == 0 or w.sum() <= 0:
            continue
        entry = {"yield": float(w.sum())}
        for leg, flavour in legs.items():
            pt, eta = frame[f"pt_{leg}"].to_numpy(), frame[f"eta_{leg}"].to_numpy()
            if flavour == "m":
                pt_c = np.clip(pt, 15.01, 119.9)
                eta_c = np.clip(np.abs(eta), 0.0, 2.399)
                for name in ("NUM_MediumID_DEN_TrackerMuons", "NUM_TightRelIso_DEN_MediumID"):
                    c = muon[name]
                    rel = vectorised(c, eta_c, pt_c, "stat") / vectorised(c, eta_c, pt_c, "nominal")
                    entry[name] = ratio(w, 1 + rel)
            else:
                pt_c = np.clip(pt, 20.01, 499.0)
                eta_c = np.clip(eta, -2.499, 2.499)
                nominal = vectorised(electron, POG_ERA_EGM[era], "sf", "RecoAbove20", eta_c, pt_c)
                for direction in ("sfup", "sfdown"):
                    entry[f"reco_{direction}"] = ratio(w, vectorised(electron, POG_ERA_EGM[era], direction, "RecoAbove20", eta_c, pt_c) / nominal)
        result[process] = entry
    return result


POG_ERA_EGM = {"2016preVFP": "2016preVFP", "2016postVFP": "2016postVFP", "2017": "2017", "2018": "2018"}


def ar_composition(config, analysis, channel):
    """Decay-mode fractions of the failing legs of the data in the anti-isolated region (FF application region)."""
    frame, w = rows(config, analysis, channel, "data", region="abcd_anti_iso")
    legs = (1, 2) if channel == "tt" else (2,)
    result = {}
    for leg in legs:
        fails = (frame[f"id_tau_vsJet_Medium_{leg}"] < 0.5).to_numpy()
        dm = frame[f"tau_decaymode_{leg}"].to_numpy().astype(int)
        total = fails.sum()
        result[leg] = {int(d): float(((dm == d) & fails).sum() / total) for d in (0, 1, 2, 10, 11)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--era", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = load_config(CONFIGS / "smrun2_binned_v7b.yaml", [f"era={args.era}"])
    analysis = load_analysis(config)
    result = {"era": args.era, "embedding": {}, "mc": {}, "leptons": {}, "ar": {}}
    for channel in ("et", "mt"):
        result["embedding"][channel] = embedding_numbers(config, analysis, args.era, channel)
    for channel in ("et", "mt", "tt"):
        result["mc"][channel] = mc_numbers(config, analysis, args.era, channel)
        result["ar"][channel] = ar_composition(config, analysis, channel)
    result["leptons"]["mt"] = lepton_numbers(config, analysis, args.era, "mt", {1: "m"})
    result["leptons"]["et"] = lepton_numbers(config, analysis, args.era, "et", {1: "e"})
    dilepton = load_config(CONFIGS / "smrun2_binned_v7b_dilepton.yaml", [f"era={args.era}"])
    dilepton_analysis = load_analysis(dilepton)
    result["leptons"]["em"] = lepton_numbers(dilepton, dilepton_analysis, args.era, "em", {1: "e", 2: "m"})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.era}.json").write_text(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
