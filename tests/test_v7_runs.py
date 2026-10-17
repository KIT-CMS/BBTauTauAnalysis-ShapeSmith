"""The v7 run YAMLs (spec 6.11): one per run type, every era with `-s era=<era>` ({era} filled by the core); the
productions P1 smrun2_tau_id_v1 and P3 smrun2_binned_v7 with the friends ff_v7 and ff_v7_w; skims and outputs under
/ceph/sdaigler/shapesmith/<tag>/; the galleries per era with the columns classic, emb, emb ff and emb ff with W."""
import json
from pathlib import Path

import pytest
from shapesmith.combination import run_card
from shapesmith.config import load_analysis, load_config
from shapesmith.model import AnalysisError

from bbtautau_shapesmith.constants import DY_NJETS_BTAG, LUMI_PB
from bbtautau_shapesmith.samples import inventory_names, inventory_path
from tests.helpers import DATABASE, ERAS, REPO
from tests.run2 import list_name

P1, P3 = "smrun2_tau_id_v1", "smrun2_binned_v7"
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
    assert {p.stem for p in (REPO / "configs").glob("*.yaml") if "smrun2" in p.stem} == set(RUNS)


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
    with pytest.raises(AnalysisError, match="no DY table for 2016preVFP, 2016postVFP"):
        load_analysis(load(name, "2016"))
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
