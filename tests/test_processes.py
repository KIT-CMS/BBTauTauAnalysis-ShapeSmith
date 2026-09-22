import pytest
from shapesmith.expressions import columns_in, selection_columns

from bbtautau_shapesmith.constants import TAU_CHANNELS, NN_CLASS_NAMES, NN_COLUMNS
from bbtautau_shapesmith.processes import SIGNAL, categories, control_variables, dilepton_processes, split_name, tau_processes
from bbtautau_shapesmith.selection import TOP_PT
from tests.helpers import branches


def test_split_names():
    assert split_name("DY", "T") == "ZTT" and split_name("DY", "L") == "ZL" and split_name("TTV", "J") == "TTVJ" and split_name("ST", "T") == "STT"


@pytest.mark.parametrize("jet_fakes,embedding", [("ff", False), ("mc", False), ("ff", True), ("mc", True)])
def test_process_table(jet_fakes, embedding):
    table = tau_processes(jet_fakes, embedding)
    names = [p.name for p in table]
    assert len(names) == len(set(names)) and len({p.key for p in table}) == len(table)
    assert names[:2] == ["data", SIGNAL]
    kinds = {p.kind for p in table}
    assert ("jet_fake" in kinds) == (jet_fakes == "mc") and ("W" in names) == (jet_fakes == "mc")
    assert ("embedding" in kinds) == embedding and ("true_tau" in kinds) == (not embedding)
    assert {"ZL", "TTL", "STL", "VVL", "TTVL", "EWK", "ggH125", "qqH125", "ttH125", "VH125"} <= set(names)
    for process in table:
        for channel in TAU_CHANNELS:
            selection = process.selection_for(channel)
            allowed = branches(channel) | ({"emb_genweight", "emb_idsel_wgt_1", "emb_idsel_wgt_2", "emb_triggersel_wgt"} if process.kind == "embedding" else set())
            assert selection_columns(selection) <= allowed, (process.name, channel, selection_columns(selection) - allowed)
    top = {p.name for p in table if "top_pt" in p.selection_for("mt").weights}
    assert top == {n for n in names if n.startswith("TT") and not n.startswith("TTV")}  # only ttbar ntuples carry topPtReweightWeight


def test_genmatch_selection_differs_per_channel():
    ztt = next(p for p in tau_processes("mc", False) if p.name == "ZTT")
    assert ztt.selection_for("mt").cuts["genmatch"] == "((gen_match_1 == 4) & (gen_match_2 == 5))"
    assert ztt.selection_for("et").cuts["genmatch"] == "((gen_match_1 == 3) & (gen_match_2 == 5))"
    assert ztt.selection_for("tt").cuts["genmatch"] == "((gen_match_1 == 5) & (gen_match_2 == 5))"


def test_categories():
    cats = categories()
    assert tuple(c.name for c in cats) == NN_CLASS_NAMES
    assert cats[0].cut == "(predicted_class == 0)" and cats[5].cut == "(predicted_class == 5)"
    assert cats[0].variable.name == "NN_score" and cats[0].variable.expr == "predicted_max_value"
    assert len(cats[0].variable.edges) == 23 and len(cats[1].variable.edges) == 21
    for c in cats:
        assert columns_in(c.cut) | columns_in(c.variable.expr) <= NN_COLUMNS


@pytest.mark.parametrize("channel", TAU_CHANNELS)
def test_control_variables_exist(channel):
    for name, variable in control_variables(nn_friend=True).items():
        assert variable.name == name
        assert columns_in(variable.expr) <= branches(channel) | NN_COLUMNS, (name, columns_in(variable.expr) - branches(channel))
        assert len(variable.edges) >= 2 and all(b > a for a, b in zip(variable.edges, variable.edges[1:]))
    assert {"m_vis", "n_bjets", "bpair_m_inv", "mass_tautaubb", "sum_deltaR_tt_bb", "NN_score"} <= set(control_variables(nn_friend=True))
    assert "NN_score" not in control_variables()


def test_dilepton_process_table():
    table = dilepton_processes()
    names = [p.name for p in table]
    assert names[:2] == ["data", SIGNAL] and len(names) == len(set(names)) and len({p.key for p in table}) == len(table)
    assert {"DY", "TT", "ST", "VV", "TTV", "W", "EWK", "ggH125", "qqH125", "ttH125", "VH125"} <= set(names)
    assert next(p for p in table if p.name == "TT").selection.weights == TOP_PT
    for process in table:
        columns = selection_columns(process.selection_for("em"))
        assert not any("gen_match" in c or "tau" in c for c in columns), (process.name, columns)
        assert columns <= branches("em")
