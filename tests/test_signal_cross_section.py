"""The unit of the limits on the cross section (spec N23): sigma(ggF HH) = r x 30.40 fb with the theory nuisances of
the cross section frozen."""
import pytest

from bbtautau_shapesmith.processes import SIGNAL, SIGNAL_XS_FB, SIGNAL_XS_NUISANCES
from tests.test_analysis import analysis_for

BR_BB, BR_TAUTAU = 0.5824, 0.06272  # the BR of the sample cross section (LHCHWG, m_H = 125.09 GeV)
SAMPLE_XS_PB = 0.002221  # of the powheg ggF HH -> bbtautau samples in the sample database


@pytest.mark.parametrize("jet_fakes,embedding", [("mc", False), ("ff", ["et", "mt"])])
def test_the_limits_are_on_sigma_ggf_hh_with_its_theory_nuisances_frozen(jet_fakes, embedding):
    analysis = analysis_for(jet_fakes, embedding)
    cross_section = analysis.signal_cross_section
    assert cross_section.quantity == "sigma(ggF HH)" and cross_section.value_fb == SIGNAL_XS_FB == 30.40
    assert cross_section.theory_nuisances == SIGNAL_XS_NUISANCES == ("QCDscale_HH", "PDF_alphas_HH")
    signal_lnn = {entry.name for entry in analysis.lnn if entry.processes == (SIGNAL,)}
    assert set(cross_section.theory_nuisances) <= signal_lnn  # the names of the cards
    assert {"BR_h_bb", "BR_h_tautau"} <= signal_lnn - set(cross_section.theory_nuisances)  # the BR stay in the limit


def test_the_cross_section_is_that_of_the_signal_sample():
    assert SIGNAL_XS_FB * 2 * BR_BB * BR_TAUTAU / 1000 == pytest.approx(SAMPLE_XS_PB, rel=1e-3)
