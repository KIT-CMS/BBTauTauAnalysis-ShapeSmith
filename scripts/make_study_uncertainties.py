"""Print the tables of src/bbtautau_shapesmith/study_uncertainties.py from the sizing JSONs of
size_v7b_uncertainties.py, the fake-factor validations of the v7b measurement and the CROWN embedding payloads.

    python scripts/make_study_uncertainties.py --sizing <dir with <era>.json> \
        --ff /ceph/sdaigler/shapesmith/smrun2_binned_v7b/output > tables.py

Tables (every lnN as (down, up) yield ratios, 4 decimals):
- EMBEDDING_TAU_ID: payload vsJet SF up/nom and down/nom per era, vsEle WP and decay-mode category (exact, flat in pT);
- EMBEDDING_TAU_ES: embedding yield ratio under the energy-scale interval per era, channel and category;
- MC_TAU: per era, channel and process class the POG tau ES per decay mode (tes_dm<N>), the vsJet components
  syst_alleras, the era part (syst_<era> and all stat components in quadrature, symmetric) and syst_TES per decay
  mode, and in tt the di-tau trigger SF per decay mode (trigger_dm<N>);
- LEPTONS: per era the muon_Z ID + isolation stat (linear sum, symmetric) and the EGM reco SF, per channel;
- FF_DM_NONCLOSURE: per variant, year, channel and leg the residual DR closure per decay-mode bin after all corrections
  (share-weighted over the processes whose decay-mode closure fails), the shift of the fake factor;
- FF_AR_SHIFT: the yield shift that shift makes on the decay-mode composition of the data AR (for the summary);
- FF_HALVES: per variant, 2016 half and channel 1 + max |combined / half - 1| of the predicted jet fakes.
"""
from __future__ import annotations

import argparse
import json
import math
import pprint
from pathlib import Path

import correctionlib

from bbtautau_shapesmith.study_uncertainties import MC_TAU_CLASSES

ERAS = ("2016preVFP", "2016postVFP", "2017", "2018")
YEARS = {"2016": ("2016preVFP", "2016postVFP"), "2017": ("2017",), "2018": ("2018",)}
CROWN = Path("/work/sdaigler/bbtautau/KingMaker/CROWN/analysis_configurations/bbtautau")
EMBEDDING_PAYLOAD = "payloads/tau_embedding/Run2-{era}-UL-NanoAODv15/DeepTau2018v2p5_id_es_embedding{era}UL.json.gz"
TOKENS = {"0": 0, "1": 1, "1011": 10}
POG_ERA_TOKEN = {"2016preVFP": "2016_preVFP", "2016postVFP": "2016_postVFP", "2017": "2017", "2018": "2018"}
DM_BINS = (0, 1, 10, 11)  # the closure bins of tau_decaymode: 0, 1 (with 2), 10, 11
# the processes whose decay-mode DR closure fails (p < 0.01 in some year, ff_v7b_files/closure_failures.md)
FF_DM_PROCESSES = {"mt": {"leading": ("ttbar",)}, "tt": {"leading": ("QCD", "ttbar"), "_subleading": ("QCD_subleading", "ttbar_subleading")}}
VARIANTS = {"fake_factors": "QCD_ttbar", "fake_factors_w": "QCD_Wjets_ttbar"}


def r4(x):
    return round(float(x), 4)


def weighted(entries, key):
    total = sum(e["yield"] for e in entries)
    return tuple(r4(sum(e["yield"] * key(e)[i] for e in entries) / total) for i in range(2))


def embedding_tau_id():
    result = {}
    for era in ERAS:
        cset = correctionlib.CorrectionSet.from_file(str(CROWN / EMBEDDING_PAYLOAD.format(era=era)))["DeepTau2018v2p5VSjet"]
        result[era] = {}
        for wp in ("VVLoose", "Tight"):
            result[era][wp] = {}
            for token, dm in TOKENS.items():
                nom, down, up = (cset.evaluate(40.0, dm, 5, "Medium", wp, syst, "dm") for syst in ("nom", "down", "up"))
                result[era][wp][token] = (r4(down / nom), r4(up / nom))
    return result


def mc_tau(sizing):
    result = {}
    for era in ERAS:
        result[era] = {}
        token = POG_ERA_TOKEN[era]
        for channel in ("et", "mt", "tt"):
            processes = sizing[era]["mc"][channel]
            result[era][channel] = {}
            for klass, names in MC_TAU_CLASSES[channel].items():
                entries = [processes[name] for name in names if name in processes and processes[name]["yield"] > 0]
                if not entries:
                    continue
                table = {}
                for dm in ("0", "1", "10", "11"):
                    table[f"tes_dm{dm}"] = weighted(entries, lambda e: (e["tes"][dm]["down"], e["tes"][dm]["up"]))
                    table[f"syst_TES_dm{dm}"] = weighted(entries, lambda e: (e["vsjet"][f"syst_TES_{token}_dm{dm}"]["down"], e["vsjet"][f"syst_TES_{token}_dm{dm}"]["up"]))
                    if channel == "tt":
                        table[f"trigger_dm{dm}"] = weighted(entries, lambda e: (e["trigger"][dm]["down"], e["trigger"][dm]["up"]))
                table["syst_alleras"] = weighted(entries, lambda e: (e["vsjet"]["syst_alleras"]["down"], e["vsjet"]["syst_alleras"]["up"]))
                era_parts = [f"syst_{token}"] + [f"stat{i}_dm{dm}" for i in (1, 2) for dm in ("0", "1", "10", "11")]

                def half_widths(e):
                    return [(e["vsjet"][c]["up"] - e["vsjet"][c]["down"]) / 2 for c in era_parts]

                total = sum(e["yield"] for e in entries)
                widths = [sum(e["yield"] * half_widths(e)[i] for e in entries) / total for i in range(len(era_parts))]
                table["era"] = r4(1 + math.sqrt(sum(w * w for w in widths)))
                result[era][channel][klass] = table
    return result


def leptons(sizing):
    result = {}
    for era in ERAS:
        result[era] = {}
        for channel, flavours in (("et", "e"), ("mt", "m"), ("em", "em")):
            entries = [e for name, e in sizing[era]["leptons"][channel].items() if e["yield"] > 0 and name != "W"]
            total = sum(e["yield"] for e in entries)
            entry = {}
            if "m" in flavours:
                entry["muon_stat"] = r4(1 + sum(e["yield"] * (e["NUM_MediumID_DEN_TrackerMuons"] - 1 + e["NUM_TightRelIso_DEN_MediumID"] - 1) for e in entries) / total)
            if "e" in flavours:
                entry["electron_reco"] = r4(1 + sum(e["yield"] * (e["reco_sfup"] - e["reco_sfdown"]) / 2 for e in entries) / total)
            result[era][channel] = entry
    return result


def embedding_tau_es(sizing):
    return {era: {ch: {token: (r4(e["es"][token]["down"]), r4(e["es"][token]["up"])) for token in TOKENS} for ch, e in sizing[era]["embedding"].items()} for era in ERAS}


def dm_bin(edges_ratio, dm):
    edges, ratios = edges_ratio
    for low, high, r in zip(edges[:-1], edges[1:], ratios):
        if low < dm < high:
            return r
    raise ValueError(dm)


def ff_dm_nonclosure(ff_dir):
    result = {}
    for directory, variant in VARIANTS.items():
        result[variant] = {}
        for year in YEARS:
            validation = json.loads((ff_dir / directory / year / "validation.json").read_text())
            result[variant][year] = {}
            for channel, legs in FF_DM_PROCESSES.items():
                closures = validation["channels"][channel]["closure"]
                predictions = validation["channels"][channel]["prediction"]
                result[variant][year][channel] = {}
                for leg, processes in legs.items():
                    preds = [p for p in predictions if p["leg"] == leg]
                    total = sum(p["prediction"] for p in preds)
                    shift = {dm: 0.0 for dm in DM_BINS}
                    for process in processes:
                        share_key = process.replace("_subleading", "")
                        share = sum(p["prediction"] * p["shares"].get(share_key, 0.0) for p in preds) / total
                        entries = [c for c in closures if c["leg"] == leg and c["process"] == process and c["variable"].startswith("tau_decaymode")]
                        weights = []
                        for c in entries:  # nj categories of the closure: split [1.5, 2.5] = 2 jets, [2.5, ...] = >= 3 jets
                            jets = "nj2" if c["split"][1] <= 2.5 and len(entries) > 1 else ("nj3p" if len(entries) > 1 else None)
                            weights.append(sum(p["prediction"] * p["shares"].get(share_key, 0.0) for p in preds if jets is None or p["category"].endswith(jets)))
                        for dm in DM_BINS:
                            r = sum(w * dm_bin((c["edges"], c["ratio"]), dm) for w, c in zip(weights, entries)) / sum(weights)
                            shift[dm] += share * (r - 1)
                    result[variant][year][channel]["1" if channel == "tt" and leg == "leading" else "2"] = {dm: round(v, 3) for dm, v in shift.items()}
    return result


def ff_ar_shift(nonclosure, sizing):
    """Yield shift of the corrected fake factors on the AR decay-mode composition (DM 2 counted as DM 1)."""
    result = {}
    for variant, years in nonclosure.items():
        result[variant] = {}
        for year, channels in years.items():
            for era in YEARS[year]:
                result[variant][era] = {}
                for channel, legs in channels.items():
                    ar = sizing[era]["ar"][channel]
                    shifts = []
                    for leg, delta in legs.items():
                        f = ar[leg]
                        shifts.append(sum(f[str(dm)] * delta[dm] for dm in DM_BINS) + f["2"] * delta[1])
                    result[variant][era][channel] = r4(sum(shifts) / len(shifts))
    return result


def ff_halves(ff_dir):
    result = {}
    for directory, variant in VARIANTS.items():
        result[variant] = {}
        for era in YEARS["2016"]:
            own = json.loads((ff_dir / directory / era / "validation.json").read_text())
            combined = json.loads((ff_dir / directory / era / "validation_with_2016_payload.json").read_text())
            result[variant][era] = {}
            for channel in ("et", "mt", "tt"):
                half = {(p["leg"], p["category"]): p["prediction"] for p in own["channels"][channel]["prediction"]}
                worst = max(abs(p["prediction"] / half[p["leg"], p["category"]] - 1) for p in combined["channels"][channel]["prediction"])
                result[variant][era][channel] = round(1 + math.ceil(worst * 1000) / 1000, 3)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizing", type=Path, required=True)
    parser.add_argument("--ff", type=Path, required=True)
    args = parser.parse_args()
    sizing = {era: json.loads((args.sizing / f"{era}.json").read_text()) for era in ERAS}
    nonclosure = ff_dm_nonclosure(args.ff)
    tables = {
        "EMBEDDING_TAU_ID": embedding_tau_id(),
        "EMBEDDING_TAU_ES": embedding_tau_es(sizing),
        "MC_TAU": mc_tau(sizing),
        "LEPTONS": leptons(sizing),
        "FF_DM_NONCLOSURE": nonclosure,
        "FF_AR_SHIFT": ff_ar_shift(nonclosure, sizing),
        "FF_HALVES": ff_halves(args.ff),
    }
    for name, table in tables.items():
        print(f"{name} = {pprint.pformat(table, width=150, sort_dicts=False)}")


if __name__ == "__main__":
    main()
