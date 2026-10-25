"""The run YAMLs of the NN categories (nn_limit_spec.md SH13, 4.2, 9): the friend productions FR1 nn_thesis2018_v1 (2018,
the thesis recipe; R3 and R4 with the background model of the thesis comparison, R5 with the final one) and FR2 nn_m0_v1
(Run 2, M0 and M0-k4; R6-R8 with the fake-factor variants ff and ff_w, N7). Each is the run it copies with the NN friend
(data under smrun2_binned_v7, MC and embedding under smrun2_binned_v7b, after the friends it chains on), nn_friend: true,
fit_categories: nn and its own skims and outputs, without ML export and gallery. The friends do not exist yet, so the
YAMLs are checked offline; every run YAML stays at the systematics stage S0 (no stage switch on)."""
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
STORE = "/store/user/sdaigler/CROWN/ntuples"
ROOT = Path("/ceph/sdaigler/shapesmith/smrun2_binned_v7b")
EXPORT_SWITCHES = ("ml_labels", "ml_variables")  # the ML export of the copied run, not of a run with the NN friend
STAGE_SWITCHES = ("ttbar_shape", "heavy_flavour", "signal_theory")
S0_PLUS = ("switches.ttbar_shape=true", "switches.heavy_flavour=true")


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
    expected["switches"].update(nn_friend=True, fit_categories="nn")
    expected["ntuples"]["friends"] = actual["ntuples"]["friends"]
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
    classes = ML_CLASSES["thesis6" if config.switches["jet_fakes"] == "mc" else "v7b5"]  # the NN column switch of SH1 follows
    for channel_name in TAU_CHANNELS:
        channel = analysis.channel(channel_name)
        assert tuple(category.name for category in channel.categories) == classes
        simulated = {p.group for p in channel.processes if p.role == "background" and p.group != EMBEDDED}
        for sample in channel.samples:  # every kind reads the NN friend, the simulated backgrounds of S0+ the b-pair flavour
            assert NN_COLUMNS <= needed_columns(channel, sample), sample.nick
            assert ("bpair_hadronFlavour_1" in needed_columns(channel, sample)) == (stage == "S0+" and sample.group in simulated), sample.nick
    varied = {v.name for channel in analysis.channels.values() for v in channel.variations}
    assert ({"CMS_htt_ttbarShapeUp", "CMS_htt_heavy_flavourUp"} <= varied) == (stage == "S0+")
    assert analysis.control_channels == (("em",) if "em" in config.channels else ())
    # the thesis comparison: the final state 'all' of et, mt and tt without em (datacards --channels et,mt,tt)
    assert _final_states(list(TAU_CHANNELS), analysis.control_channels)["all"] == list(TAU_CHANNELS)


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
