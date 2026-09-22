import pytest
from shapesmith.expressions import columns_in, selection_columns

from bbtautau_shapesmith.constants import TAU_CHANNELS, NN_COLUMNS
from bbtautau_shapesmith.processes import control_variables, dilepton_processes, tau_processes
from tests.helpers import branches


@pytest.mark.parametrize("jet_fakes,embedding", [("ff", False), ("mc", False), ("ff", True), ("mc", True)])
def test_process_table(jet_fakes, embedding):
    table = tau_processes(jet_fakes, embedding)
    names = [p.name for p in table]
    assert len(names) == len(set(names)) and len({p.key for p in table}) == len(table)
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
    assert len(names) == len(set(names)) and len({p.key for p in table}) == len(table)
    assert {"DY", "TT", "ST", "VV", "TTV", "W", "EWK", "ggH125", "qqH125", "ttH125", "VH125"} <= set(names)
    for process in table:
        columns = selection_columns(process.selection_for("em"))
        assert not any("gen_match" in c or "tau" in c for c in columns), (process.name, columns)
        assert columns <= branches("em")
