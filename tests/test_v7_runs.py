"""The v7 run YAMLs (spec 6.11): one per run type, every era with `-s era=<era>` ({era} filled by the core); the
productions P1 smrun2_tau_id_v1 and P3 smrun2_binned_v7 with the friends ff_v7 and ff_v7_w; skims and outputs under
/ceph/sdaigler/shapesmith/<tag>/; the galleries per era with the columns classic, emb, emb ff and emb ff with W.
The re-production of 2026-10-18 (MET fix, POG muon trigger SF on MC): a copy of every run YAML that reads the MC and the
embedding of P3 from smrun2_binned_v7b and the MC of P1 from smrun2_tau_id_v2 (ntuples.bases), with its own skims,
outputs and galleries under the new tag. The decision of 2026-10-19 (E33: the tt-embedding closure failed in every era):
the v7b runs with the embedding take the genuine taus of tt from MC, switches.embedding [et, mt] for [et, mt, tt]. The
v7b galleries of the eras are one site each (web.site, 2026-10-19): one page with an era switch and a title without the era."""
import json
from pathlib import Path

import pytest
from shapesmith.combination import run_card
from shapesmith.config import FriendConfig, load_analysis, load_config
from shapesmith.model import AnalysisError
from shapesmith.skim import needed_columns

from bbtautau_shapesmith import weights
from bbtautau_shapesmith.constants import DY_NJETS_BTAG, LUMI_PB
from bbtautau_shapesmith.samples import inventory_names, inventory_path
from tests.helpers import DATABASE, ERAS, REPO
from tests.run2 import list_name
from tests.test_nn_runs import NN_FRIEND_RUNS

P1, P3 = "smrun2_tau_id_v1", "smrun2_binned_v7"
REPRODUCTION = {P3: ("smrun2_binned_v7b", ("mc", "embedding")), P1: ("smrun2_tau_id_v2", ("mc",))}  # tag, re-produced kinds
STORE = "/store/user/sdaigler/CROWN/ntuples"
RUNS = {  # run YAML -> (production, gallery column or measurement name)
    "smrun2_binned_v7": (P3, "emb"),
    "smrun2_binned_v7_noemb": (P3, "classic"),
    "smrun2_binned_v7_dilepton": (P3, "classic"),
    "smrun2_binned_v7_ff": (P3, "emb_ff"),
    "smrun2_binned_v7_ff_w": (P3, "emb_ff_w"),
    "ff_smrun2_binned_v7": (P3, "fake_factors"),
    "ff_smrun2_binned_v7_w": (P3, "fake_factors_w"),
    "smrun2_tau_id_v1_id_es": (P1, "tau_id_es"),
    "smrun2_tau_id_v1_trigger": (P1, "tau_trigger"),
}
MEASUREMENTS = ("ff_smrun2_binned_v7", "ff_smrun2_binned_v7_w", "smrun2_tau_id_v1_id_es", "smrun2_tau_id_v1_trigger")
COLUMNS = {"classic": "classic", "emb": "emb", "emb_ff": "emb ff", "emb_ff_w": "emb ff with W"}
REPRODUCED = {name: name.replace(production, REPRODUCTION[production][0]) for name, (production, _) in RUNS.items()}  # v7 -> v7b
TT_FROM_MC = ("smrun2_binned_v7", "smrun2_binned_v7_ff", "smrun2_binned_v7_ff_w", "ff_smrun2_binned_v7", "ff_smrun2_binned_v7_w")  # E33
TT_FROM_MC_TEXT = ("in et, mt and tt", "in et and mt (tt: MC)")  # the web description of the FF measurement says so


def tt_from_mc(name: str, description: str) -> str:
    """The v7b web description of a run with tt from MC (E33): the FF measurement names the subtracted channels; the
    embedded columns of the control site hold et and mt (emb ff also em) and say where tt is, as in v6: classic, or the
    column ff (the tt of the FF run, published with --channels tt --variant ff, README)."""
    if name.startswith("ff_"):
        return description.replace(*TT_FROM_MC_TEXT)
    see = "classic" if name == "smrun2_binned_v7" else "ff"
    return description.replace(" in et, mt and tt,", ",").removesuffix(".") + f"; et and mt only (tt has no embedding, see {see})."
# the v7b fake-factor runs carry the uncertainties sized in the v7b studies (switch study_uncertainties, 2026-10-19)
STUDY_UNCERTAINTIES = ("smrun2_binned_v7_ff", "smrun2_binned_v7_ff_w")
# the v7b tau runs read the FastMTT friend of P4 (data under v7, MC and embedding under v7b) for the control variables
# m, pt, eta and phi_fastmtt (switch fastmtt_friend, 2026-10-20); the v7 runs never read it
FASTMTT = ("smrun2_binned_v7", "smrun2_binned_v7_noemb", "smrun2_binned_v7_ff", "smrun2_binned_v7_ff_w")
RENAMED = {P3: (("ff_v7", "ff_v7b"),), P1: ()}  # the FF friends of v7b get their own names (ff_v7b, ff_v7b_w): P5 of v7 and v7b never overwrite each other
# the NN phase (nn_limit_spec.md 3.1, 2026-10-25): the v7b classic run exports the training data of the thesis
# comparison (X10, label profile thesis6) with the variables its skims hold, and the v7b FF run has a copy with the
# export of the profile v7b5 and its own skims (the re-skim of X8)
NN_EXPORT = {"smrun2_binned_v7_noemb": ("thesis6", {"ml_labels": "thesis6", "ml_variables": ["thesis", "objects", "kinematics"]})}
NN_RUNS = {"smrun2_binned_v7b_ff_nn": "smrun2_binned_v7b_ff"}  # NN run -> the run it copies


def load(name: str, era: str, *overrides: str):
    return load_config(REPO / "configs" / f"{name}.yaml", [f"era={era}", f"sample_database={DATABASE}", *overrides])


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from strings(item)


def test_one_run_yaml_per_run_type():
    assert {p.stem for p in (REPO / "configs").glob("*.yaml") if "smrun2" in p.stem} == set(RUNS) | set(REPRODUCED.values()) | set(NN_RUNS) | set(NN_FRIEND_RUNS)


def moved(value, old: str, new: str, renamed=()):
    """Every string of a dumped configuration with the tag `old` replaced by `new`, then the `renamed` (old, new) pairs."""
    if isinstance(value, str):
        value = value.replace(old, new)
        for before, after in renamed:
            value = value.replace(before, after)
        return value
    if isinstance(value, dict):
        return {key: moved(item, old, new, renamed) for key, item in value.items()}
    if isinstance(value, list):
        return [moved(item, old, new, renamed) for item in value]
    return value


@pytest.mark.parametrize("era", ERAS)
@pytest.mark.parametrize("name", RUNS)
def test_the_reproduction_reads_its_kinds_from_the_new_tag(name, era):
    """The copy of a run YAML: the data (P1: and the embedding) from the first production, the re-produced kinds and their
    friends from the new tag; skims, outputs, ML folds, galleries and payload paths under the new tag; the rest is the run
    of the first production (the about text aside, the recoil eras of the tau-ID measurement and, in the runs with the
    embedding, tt from MC: switches.embedding [et, mt] and the matching web description; in the fake-factor runs the
    uncertainties of the v7b studies, switches.study_uncertainties; the galleries as eras of a site)."""
    production, _ = RUNS[name]
    tag, kinds = REPRODUCTION[production]
    first, again = load(name, era), load(REPRODUCED[name], era)
    assert again.ntuples.server == first.ntuples.server and again.ntuples.base == first.ntuples.base == f"{STORE}/{production}/CROWNRun"
    assert again.ntuples.bases == {kind: f"{STORE}/{tag}/CROWNRun" for kind in kinds} and not first.ntuples.bases
    split = []
    for friend in first.ntuples.friends:
        assert set(friend.applies_to) == {"mc", "data", "embedding"}
        base = moved(friend.base, "", "", RENAMED[production])
        split += [friend.model_copy(update={"base": base, "applies_to": tuple(k for k in ("data", "mc", "embedding") if k not in kinds)}),
                  friend.model_copy(update={"base": base.replace(f"/{production}/", f"/{tag}/"), "applies_to": kinds})]
    if name in FASTMTT:
        assert not any("fastmtt" in friend.base for friend in first.ntuples.friends)
        split += [FriendConfig(base=f"{STORE}/{production}/CROWNFriend/fastmtt", applies_to=("data",), channels=("et", "mt", "tt")),
                  FriendConfig(base=f"{STORE}/{tag}/CROWNFriend/fastmtt", applies_to=kinds, channels=("et", "mt", "tt"))]
    assert again.ntuples.friends == split
    for path in (again.skim_dir, again.output_dir, *([again.ml_dir] if again.ml_dir else [])):
        assert str(path).startswith(f"/ceph/sdaigler/shapesmith/{tag}/"), path
    rest = [{key: value for key, value in json.loads(c.model_dump_json()).items() if key != "ntuples"} for c in (first, again)]
    for dump in rest:
        if dump.get("web"):
            dump["web"].pop("about")
        recoil = dump["switches"].pop("recoil_eras", None)
        assert recoil is None or name.endswith("_id_es")
    expected = moved(rest[0], production, tag, RENAMED[production])
    if expected.get("web"):  # the gallery of the era is one era of the site, whose page names no era
        assert "site" not in rest[0]["web"]
        expected["web"]["site"] = str(Path(expected["web"]["dir"]).parent)
        if expected["web"].get("title"):
            expected["web"]["title"] = expected["web"]["title"].replace(f" {era} ", " ")
    if name in TT_FROM_MC:
        assert first.switches["embedding"] == ["et", "mt", "tt"] and again.switches["embedding"] == ["et", "mt"]
        expected["switches"]["embedding"] = ["et", "mt"]
        if expected.get("web"):  # the runs whose description names the embedded channels
            expected["web"]["description"] = tt_from_mc(name, expected["web"]["description"])
    else:
        assert again.switches.get("embedding") == first.switches.get("embedding")
    if name in STUDY_UNCERTAINTIES:
        assert "study_uncertainties" not in first.switches and again.switches["study_uncertainties"] is True
        expected["switches"]["study_uncertainties"] = True
        # the fake-factor shifts symmetrised where one-sided or strongly asymmetric (2026-10-19)
        assert "symmetrise_ff_shifts" not in first.switches and again.switches["symmetrise_ff_shifts"] is True
        expected["switches"]["symmetrise_ff_shifts"] = True
    else:
        assert "study_uncertainties" not in again.switches and "symmetrise_ff_shifts" not in again.switches
    if name in FASTMTT:
        assert "fastmtt_friend" not in first.switches and again.switches["fastmtt_friend"] is True
        expected["switches"]["fastmtt_friend"] = True
    else:
        assert "fastmtt_friend" not in again.switches
    if name in NN_EXPORT:
        profile, switches = NN_EXPORT[name]
        assert first.ml_dir is None and again.ml_dir == Path(f"/ceph/sdaigler/shapesmith/{tag}/ml/{profile}/{era}")
        expected["ml_dir"] = str(again.ml_dir)
        expected["switches"].update(switches)
    else:
        assert "ml_labels" not in again.switches
    assert rest[1] == expected
    if name.endswith("_id_es"):  # the MET fix: the recoil shifts of DY and W in 2018 only; v1 has them in every era
        assert first.switches["recoil_eras"] == list(ERAS) and again.switches["recoil_eras"] == ["2018"]
    overrides = ["switches.emb_es_payload=null"] if name.endswith("_trigger") else []
    if again.switches.get("dy_correction") == "njets_btag" and era not in DY_NJETS_BTAG:
        overrides.append("switches.dy_correction=none")
    analysis = load_analysis(load(REPRODUCED[name], era, *overrides))
    assert analysis.era == era and analysis.lumi_pb == LUMI_PB[era]


@pytest.mark.parametrize("era", ERAS)
@pytest.mark.parametrize("name", RUNS)
def test_every_run_yaml_serves_every_era(name, era):
    production, _ = RUNS[name]
    config = load(name, era)
    assert config.era == era and not [s for s in strings(json.loads(config.model_dump_json())) if "{era}" in s]
    assert config.ntuples.server == "root://cmsdcache-kit-disk.gridka.de" and config.ntuples.base == f"/store/user/sdaigler/CROWN/ntuples/{production}/CROWNRun"
    assert all(friend.base.startswith(f"/store/user/sdaigler/CROWN/ntuples/{production}/CROWNFriend/") for friend in config.ntuples.friends)
    for path in (config.skim_dir, config.output_dir, *([config.ml_dir] if config.ml_dir else [])):
        assert str(path).startswith(f"/ceph/sdaigler/shapesmith/{production}/"), path
    assert config.skim_dir.name == era and (config.output_dir.name == "output" if name in MEASUREMENTS else config.output_dir.name == era)
    assert config.workers <= 24 and config.log_level == "DEBUG"
    for sample_list in config.switches["sample_lists"]:
        assert inventory_path(sample_list).name.startswith(f"sm_{era}_")
    overrides = []
    if config.switches.get("dy_correction") == "njets_btag" and era not in DY_NJETS_BTAG:
        with pytest.raises(AnalysisError, match=f"no DY table for {era}"):
            load_analysis(config)
        overrides.append("switches.dy_correction=none")  # until the dilepton run of the era gives its table (spec 6.10)
    if name == "smrun2_tau_id_v1_trigger":
        assert config.switches["emb_es_payload"] == f"/ceph/sdaigler/shapesmith/{P1}/output/tau_id_es/{era}/DeepTau2018v2p5_id_es_embedding{era}UL.json.gz"
        assert Path(config.switches["reference_payload"]).name == "tau.json.gz" and f"Run2-{era}-UL-NanoAODv15" in config.switches["reference_payload"]
        assert config.switches["kit_reference_payload"].endswith(f"/CROWN/data/embedding/tau_trigger{era}_UL.json.gz")
        overrides.append("switches.emb_es_payload=null")  # the payload of the tau-ID measurement comes first
    analysis = load_analysis(load(name, era, *overrides))
    assert analysis.era == era and analysis.lumi_pb == LUMI_PB[era]


@pytest.mark.parametrize("name", ["ff_smrun2_binned_v7", "ff_smrun2_binned_v7_w"])
def test_the_2016_fake_factors_are_measured_on_both_halves(name):
    """-s era=2016: the lists sm_2016_binned and sm_2016_embedding stand for those of both halves (E18)."""
    config = load(name, "2016", "switches.dy_correction=none")
    assert config.skim_dir == Path(f"/ceph/sdaigler/shapesmith/{P3}/skims/ff_measurement/2016")
    assert [n for sample_list in config.switches["sample_lists"] for n in inventory_names(sample_list, "2016")] == [
        list_name(kind, half) for kind in ("binned", "embedding") for half in ("2016preVFP", "2016postVFP")]
    analysis = load_analysis(config)
    assert analysis.era == "2016" and {s.era for s in analysis.channel("mt").samples} == {"2016preVFP", "2016postVFP"}
    # the DY of both halves carries the luminosity-weighted mean of their tables (weights.dy_table)
    dy = [p for p in load_analysis(load(name, "2016")).channel("mt").processes if p.group == "DY"]
    assert dy and all(p.selection.weights["dy_correction"] == weights.dy_njets_btag_weight("2016") for p in dy)
    assert inventory_names("sm_2017_binned", "2016") == ("sm_2017_binned",) and inventory_names("sm_2016_binned", "2017") == ("sm_2016_binned",)


def test_the_galleries_of_an_era_and_their_columns():
    for era in ERAS:
        columns = {}
        for name, (production, column) in RUNS.items():
            web = load(name, era).web
            if name in MEASUREMENTS:
                assert web is None or web.dir == Path(f"/web/sdaigler/public_html/ff_{P3}/{era}") and web.variant == {"fake_factors": "ff_v7", "fake_factors_w": "ff_v7_w"}[column]
                continue
            assert web.dir == Path(f"/web/sdaigler/public_html/{P3}/{era}") and web.variant == column and web.label == COLUMNS[column]
            columns.setdefault(column, []).append(name)
            assert "{era}" not in (web.title or "") + (web.about or "") + (web.description or "")
        assert columns == {"emb": ["smrun2_binned_v7"], "classic": ["smrun2_binned_v7_noemb", "smrun2_binned_v7_dilepton"],
                           "emb_ff": ["smrun2_binned_v7_ff"], "emb_ff_w": ["smrun2_binned_v7_ff_w"]}
        assert load("smrun2_binned_v7", era).web.title == f"smrun2_binned_v7 {era} · control plots"
        assert load("smrun2_binned_v7_noemb", era).web.description == load("smrun2_binned_v7_dilepton", era).web.description


def test_the_v7b_galleries_are_the_eras_of_two_sites():
    for era in ERAS:
        for name in RUNS:
            web = load(REPRODUCED[name], era).web
            if web is None:
                continue
            site = "ff_smrun2_binned_v7b" if name in MEASUREMENTS else "smrun2_binned_v7b"
            assert web.site == Path(f"/web/sdaigler/public_html/{site}") and web.dir == web.site / era
            assert era not in (web.title or "")
        assert load("smrun2_binned_v7b", era).web.title == "smrun2_binned_v7b · control plots"
        assert load("ff_smrun2_binned_v7b", era).web.title == "smrun2_binned_v7b · fake factors"


def test_the_runs_of_a_production_fit_together():
    for era in ERAS:
        runs = {name: load(name, era) for name in RUNS}
        emb, noemb, ff, ff_w = (runs[n] for n in ("smrun2_binned_v7", "smrun2_binned_v7_noemb", "smrun2_binned_v7_ff", "smrun2_binned_v7_ff_w"))
        assert noemb.switches == emb.switches | {"embedding": False} and noemb.skim_dir == emb.skim_dir
        assert ff_w.switches == ff.switches | {"ff_processes": ["QCD", "Wjets", "ttbar"]} and ff.switches["ff_processes"] == ["QCD", "ttbar"]
        assert [f.base.rsplit("/", 1)[1] for f in ff.ntuples.friends] == ["ff_v7"] and [f.base.rsplit("/", 1)[1] for f in ff_w.ntuples.friends] == ["ff_v7_w"]
        assert ff.channels == ff_w.channels == ["et", "mt", "tt", "em"] and all(f.channels == ("et", "mt", "tt") for f in ff.ntuples.friends + ff_w.ntuples.friends)
        measurement, measurement_w = runs["ff_smrun2_binned_v7"], runs["ff_smrun2_binned_v7_w"]
        assert measurement_w.switches == measurement.switches | {"ff_processes": ["QCD", "Wjets", "ttbar"]} and measurement.skim_dir == measurement_w.skim_dir
        assert measurement.switches["embedding"] == ff.switches["embedding"] == emb.switches["embedding"] and measurement.switches["ff_processes"] == ff.switches["ff_processes"]
        assert all(run.switches.get("crown_shifts") is False for run in (emb, noemb, ff, ff_w))  # nominal only (E34)
        tau_runs = [n for n in RUNS if n not in MEASUREMENTS]
        assert len({runs[n].output_dir for n in tau_runs}) == len(tau_runs)
        assert len({runs[n].skim_dir for n in RUNS}) == len(RUNS) - 2  # noemb shares the emb skim, the two FF measurements one skim
        assert runs["smrun2_tau_id_v1_trigger"].output_dir == runs["smrun2_tau_id_v1_id_es"].output_dir  # tau_trigger/<era>, tau_id_es/<era>
    cards = {run_card(load("smrun2_binned_v7_ff", era), "all") for era in ERAS}  # combine --era: one card per era
    assert cards == {Path(f"/ceph/sdaigler/shapesmith/{P3}/output/tau_ff/{era}/datacards/all/combined.txt") for era in ERAS}


@pytest.mark.parametrize("era", ERAS)
def test_the_nn_export_runs(era):
    """The v7b5 run is the v7b FF run with the label profile v7b5, every variable set and its own skims, outputs and
    ML directory and no gallery; the thesis6 export of the classic run needs no column its stored skims lack."""
    for name, copied in NN_RUNS.items():
        nn, ff = load(name, era), load(copied, era)
        dumps = [{key: value for key, value in json.loads(c.model_dump_json()).items() if key not in ("skim_dir", "output_dir", "ml_dir", "web")} for c in (nn, ff)]
        assert dumps[0]["switches"].pop("ml_labels") == "v7b5" and dumps[0] == dumps[1] and nn.web is None
        root = "/ceph/sdaigler/shapesmith/smrun2_binned_v7b"
        assert (nn.skim_dir, nn.output_dir, nn.ml_dir) == tuple(Path(f"{root}/{kind}/{era}") for kind in ("skims/tau_ff_nn", "output/tau_ff_nn", "ml/v7b5"))
        overrides = [] if era in DY_NJETS_BTAG else ["switches.dy_correction=none"]
        ml = load_analysis(load(name, era, *overrides)).ml
        assert ml.profile == "v7b5" and {"bpair_mass_1", "jtag_value_2", "bpair_hadronFlavour_1", "jet_hadronFlavour_2"} <= set(ml.variables)
    config = load("smrun2_binned_v7b_noemb", era, *([] if era in DY_NJETS_BTAG else ["switches.dy_correction=none"]))
    analysis = load_analysis(config)
    assert analysis.ml.profile == "thesis6" and config.skim_dir == load("smrun2_binned_v7b", era).skim_dir
    for channel in analysis.channels.values():
        for sample in channel.samples:
            assert needed_columns(channel, sample, analysis.ml) == needed_columns(channel, sample), sample.nick
