"""The run YAMLs of the NN categories (nn_limit_spec.md SH13, 4.2, 9): the friend productions FR1 nn_thesis2018_v1 (2018,
the thesis recipe; R3 and R4 with the background model of the thesis comparison, R5 with the final one) and FR2 nn_m0_v1
(Run 2, M0 and M0-k4; R6-R8 with the fake-factor variants ff and ff_w, N7). Each is the run it copies with the NN friend
(data under smrun2_binned_v7, MC and embedding under smrun2_binned_v7b, after the friends it chains on), nn_friend: true,
fit_categories: nn, the model of the run (nn_prefix) among the models of its friend (nn_friend_prefixes, whose columns
the skims keep), the categories and binning of the thesis (R3-R5) or K1 with the data-blind binning (R6-R8), and its own
skims and outputs, without ML export and gallery, and the minimizer fallback of the validation runs (a failed fit to the
data again with Migrad at strategy 0). The friends do not exist yet, so the YAMLs are checked offline; every run YAML
stays at the systematics stage S0 (no stage switch on)."""
import json
from pathlib import Path

import pytest
import yaml
from shapesmith.cli import _final_states
from shapesmith.config import FriendConfig, load_analysis, load_config
from shapesmith.skim import needed_columns

from bbtautau_shapesmith.constants import NN_COLUMNS, TAU_CHANNELS
from bbtautau_shapesmith.ml import ML_CLASSES
from bbtautau_shapesmith.processes import EMBEDDED
from tests.helpers import DATABASE, ERAS, REPO

NN_FRIEND_RUNS = {  # run YAML -> (friend tag, the run it copies, the eras of the friend)
    "smrun2_binned_v7b_nn_thesis2018_v1": ("nn_thesis2018_v1", "smrun2_binned_v7b_noemb", ("2018",)),
    "smrun2_binned_v7b_ff_nn_thesis2018_v1": ("nn_thesis2018_v1", "smrun2_binned_v7b_ff", ("2018",)),
    "smrun2_binned_v7b_ff_nn_m0_v1": ("nn_m0_v1", "smrun2_binned_v7b_ff", ERAS),
    "smrun2_binned_v7b_ff_w_nn_m0_v1": ("nn_m0_v1", "smrun2_binned_v7b_ff_w", ERAS),
}
RUNS = [(name, era) for name, (_, _, eras) in NN_FRIEND_RUNS.items() for era in eras]
FR1_PREFIXES = ["nn_thesis_nodesm_", "nn_thesis_powheg_"]  # the node_SM (R3) and the powheg (R4, R5) model
FR2_PREFIXES = ["nn_m0_", "nn_m0k4_"]  # M0 and M0-k4 (R8, N28)
THESIS = {"nn_categories": "thesis", "nn_binning": "thesis"}  # spec 4.1, 6.3 no. 8
NN_SWITCHES = {  # run YAML -> its NN switches (SH1-SH3): the model of the run and every model of its friend
    "smrun2_binned_v7b_nn_thesis2018_v1": {"nn_prefix": "nn_thesis_powheg_", "nn_friend_prefixes": FR1_PREFIXES, **THESIS},
    "smrun2_binned_v7b_ff_nn_thesis2018_v1": {"nn_prefix": "nn_thesis_powheg_", "nn_friend_prefixes": FR1_PREFIXES, "nn_classes": list(ML_CLASSES["thesis6"]), **THESIS},
    "smrun2_binned_v7b_ff_nn_m0_v1": {"nn_prefix": "nn_m0_", "nn_friend_prefixes": FR2_PREFIXES, "nn_categories": "k1", "nn_binning": "data_blind"},
    "smrun2_binned_v7b_ff_w_nn_m0_v1": {"nn_prefix": "nn_m0_", "nn_friend_prefixes": FR2_PREFIXES, "nn_categories": "k1", "nn_binning": "data_blind"},
}
STORE = "/store/user/sdaigler/CROWN/ntuples"
ROOT = Path("/ceph/sdaigler/shapesmith/smrun2_binned_v7b")
EXPORT_SWITCHES = ("ml_labels", "ml_variables")  # the ML export of the copied run, not of a run with the NN friend
STAGE_SWITCHES = ("ttbar_shape", "heavy_flavour", "signal_theory")
S0_PLUS = ("switches.ttbar_shape=true", "switches.heavy_flavour=true")
FALLBACK = "--cminFallbackAlgo Minuit2,Migrad,0:0.1"  # combine.minimizer of the NN runs and of the validation runs


def load(name: str, era: str, *overrides: str):
    return load_config(REPO / "configs" / f"{name}.yaml", [f"era={era}", f"sample_database={DATABASE}", *overrides])


def dump(config) -> dict:
    return {key: value for key, value in json.loads(config.model_dump_json()).items() if key not in ("skim_dir", "output_dir", "ml_dir", "web")}


@pytest.mark.parametrize("name,era", RUNS)
def test_each_nn_run_is_its_run_with_the_nn_friend(name, era):
    tag, copied, _ = NN_FRIEND_RUNS[name]
    nn, run = load(name, era), load(copied, era)
    assert nn.ntuples.friends == run.ntuples.friends + [
        FriendConfig(base=f"{STORE}/smrun2_binned_v7/CROWNFriend/{tag}", applies_to=("data",), channels=TAU_CHANNELS),
        FriendConfig(base=f"{STORE}/smrun2_binned_v7b/CROWNFriend/{tag}", applies_to=("mc", "embedding"), channels=TAU_CHANNELS),
    ]
    assert any(friend.base.endswith("/fastmtt") for friend in run.ntuples.friends)  # the NN friend chains on fastmtt
    expected, actual = dump(run), dump(nn)
    for key in EXPORT_SWITCHES:
        expected["switches"].pop(key, None)
    expected["switches"].update(nn_friend=True, fit_categories="nn", **NN_SWITCHES[name])
    expected["ntuples"]["friends"] = actual["ntuples"]["friends"]
    assert run.combine.minimizer == "" and nn.combine.minimizer == FALLBACK
    expected["combine"]["minimizer"] = FALLBACK
    assert actual == expected
    assert nn.ml_dir is None and nn.web is None and nn.workers <= 16


@pytest.mark.parametrize("name,era", RUNS)
def test_the_nn_runs_have_their_own_skims_and_outputs(name, era):
    nn = load(name, era)
    run = name.removeprefix("smrun2_binned_v7b_")
    skims = {"nn_thesis2018_v1": "tau_control_nn_thesis2018_v1"}.get(run, f"tau_{run}")
    assert (nn.skim_dir, nn.output_dir) == (ROOT / "skims" / skims / era, ROOT / "output" / f"tau_{run}" / era)
    others = [path.stem for path in (REPO / "configs").glob("smrun2_binned_v7b*.yaml") if path.stem != name]
    assert nn.skim_dir not in {load(other, era).skim_dir for other in others}  # a new friend: never the skims of another run
    assert nn.output_dir not in {load(other, era).output_dir for other in others}


@pytest.mark.parametrize("name,era", RUNS)
@pytest.mark.parametrize("stage", ["S0", "S0+"])
def test_the_nn_runs_build_with_the_nn_categories(name, era, stage):
    config = load(name, era, *(S0_PLUS if stage == "S0+" else ()))
    analysis = load_analysis(config)
    switches = NN_SWITCHES[name]
    classes = ML_CLASSES["thesis6" if name.endswith("thesis2018_v1") else "v7b5"]  # R5: the thesis model on the ff background model
    prefix = switches["nn_prefix"]
    for channel_name in TAU_CHANNELS:
        channel = analysis.channel(channel_name)
        assert [category.name for category in channel.categories] == list(classes)
        assert [category.cut for category in channel.categories] == [
            f"({prefix}predicted_class == {index})" + ("" if switches["nn_categories"] == "thesis" else " & (n_bjets < 3)") for index in range(len(classes))
        ]
        binning = channel.categories[0].variable.edges
        assert channel.categories[0].variable.expr == f"{prefix}predicted_max_value" and (binning.thesis is None) == (switches["nn_binning"] == "data_blind")
        simulated = {p.group for p in channel.processes if p.role == "background" and p.group != EMBEDDED}
        for sample in channel.samples:  # every kind reads the NN friend (all its models), the simulated backgrounds of S0+ the b-pair flavour
            columns = needed_columns(channel, sample)
            assert {f"{p}{c}" for p in switches["nn_friend_prefixes"] for c in NN_COLUMNS} <= columns and not NN_COLUMNS & columns, sample.nick
            assert ("bpair_hadronFlavour_1" in columns) == (stage == "S0+" and sample.group in simulated), sample.nick
    varied = {v.name for channel in analysis.channels.values() for v in channel.variations}
    assert ({"CMS_htt_ttbarShapeUp", "CMS_htt_heavy_flavourUp"} <= varied) == (stage == "S0+")
    assert analysis.control_channels == (("em",) if "em" in config.channels else ())
    # the thesis comparison: the final state 'all' of et, mt and tt without em (datacards --channels et,mt,tt)
    assert _final_states(list(TAU_CHANNELS), analysis.control_channels)["all"] == list(TAU_CHANNELS)


@pytest.mark.parametrize("name", sorted(NN_SWITCHES))
def test_the_runs_of_every_model_of_a_friend_share_the_skims(name):
    """The skims of a run hold the columns of every model of its friend: the run of another model (-s
    switches.nn_prefix=..., as the YAML headers name them; R3 also with its node_SM signal) needs exactly the columns of
    the same samples, so the stored skims serve it without skim --force."""
    era = NN_FRIEND_RUNS[name][2][-1]
    run = load_analysis(load(name, era))
    for prefix in NN_SWITCHES[name]["nn_friend_prefixes"]:
        signal = ("switches.signal_sample=node_sm",) if prefix == "nn_thesis_nodesm_" and name == "smrun2_binned_v7b_nn_thesis2018_v1" else ()
        other = load_analysis(load(name, era, f"switches.nn_prefix={prefix}", *signal))
        for channel_name in TAU_CHANNELS:
            mine = {sample.nick: sample for sample in run.channel(channel_name).samples}
            shared = [sample for sample in other.channel(channel_name).samples if sample.nick in mine]
            assert len(shared) >= len(mine) - 1  # R3 swaps the signal sample only
            for sample in shared:
                assert needed_columns(other.channel(channel_name), sample) == needed_columns(run.channel(channel_name), mine[sample.nick]), (prefix, sample.nick)
            assert other.channel(channel_name).categories[0].cut.startswith(f"({prefix}predicted_class == 0)")


def test_the_thesis_runs_are_2018_only():
    for name, (tag, _, eras) in NN_FRIEND_RUNS.items():
        assert (eras == ("2018",)) == (tag == "nn_thesis2018_v1")
        assert yaml.safe_load((REPO / "configs" / f"{name}.yaml").read_text())["era"] == "2018"
    assert load("smrun2_binned_v7b_nn_thesis2018_v1", "2018").channels == list(TAU_CHANNELS)  # as the thesis: no em


def test_every_run_yaml_stays_at_the_systematics_stage_s0():
    """The stage switches (S0+, S1) are on only by -s overrides: every run YAML builds as before."""
    for path in sorted((REPO / "configs").glob("*.yaml")):
        switches = (yaml.safe_load(path.read_text()) or {}).get("switches", {})
        assert not set(STAGE_SWITCHES) & set(switches), path.name
