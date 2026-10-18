"""Correlation of the datacard nuisances between the eras (v7 spec 6.7, E21, E22, K4, K6).

Every nuisance name is built as nuisance(name, family, era): the family's scheme decides the era token.
- K: correlated over all eras, no token;
- E: per era, the 2016 halves separate (E22); the eras whose TAU POG payload is a byte-identical copy of the 2018 one
  (E12: VSe 2016, VSmu 2016 and 2017, the e->tau energy scale 2016) take the 2018 token, so that the combination does not
  treat a copied uncertainty as independent (K6); the year-specific JES sources take the token of the JME payload
  (2016APV, 2016);
- J: per year, the 2016 halves shared (one 2016 measurement or file).
The lumi uncertainty has its own scheme (systematics.LUMI, spec 6.6). In v7 (nominal only, E34) the datacards carry
the lnN and the weight variations of the nominal tree; the families of CROWN shifts (TES, JES, JER, MET, recoil, the
tau vsJet/VSe/VSmu components) follow with the shift production, with the names this module gives them. Until then the
flat lnN eff_t_vsLep stands for VSe and VSmu together: the 2016 eras, where both are 2018 copies, take the 2018 token;
2017 keeps its own, since its VSe payload is its own (a flat lnN cannot share only its VSmu part).
"""
from __future__ import annotations

from bbtautau_shapesmith.constants import ERAS, YEAR

K, E, J = "K", "E", "J"
FAMILIES = {
    "tau_vsjet_alleras": K,  # MC tau vsJet SF component syst_alleras (POG)
    "tau_mc": E,  # MC tau ID (stat, syst_<era>, syst_TES_<era>_dm<DM>), TES and mu->tau ES per DM, the flat MC tau-ID lnN
    "tau_vse": E,  # VSe SF of e->tau fakes (POG)
    "tau_vsmu": E,  # VSmu SF of mu->tau fakes (POG)
    "tau_vslep": E,  # the flat VSe + VSmu lnN of the nominal v7 cards (eff_t_vsLep), the stand-in for both components
    "tau_e_es": E,  # e->tau energy scale (POG)
    "embedding": E,  # embedding tau ID/ES, embedding normalisation, ttbar contamination of the embedded sample
    "trigger": E,  # e, mu and di-tau triggers, MC and embedding
    "lepton_id": K,  # e/mu ID and isolation (lnN 2 %), the systematic part of muon_Z (K4)
    "muon_id_stat": E,  # the statistical part of muon_Z (K4)
    "electron_reco": K,  # electron reconstruction, uncorrelated with the ID (EGM, K4)
    "electron_es": K,  # electron energy scale and resolution (EGM, K4)
    "jes_correlated": K,  # the regrouped JES sources correlated between the years
    "jes_year": E,  # the year-specific regrouped JES sources
    "jer": E,
    "met_unclustered": E,
    "recoil": J,  # one recoil file per year; since the MET fix (constants.RECOIL_ERAS) only 2018 has a recoil correction
    "prefiring": K,  # one CMS_prefiring for 2016-2018 (E11)
    "pileup": K,  # E21
    "btag_correlated": K,
    "btag_uncorrelated": E,
    "fake_factors": J,  # FF shifts and jetFakesNorm per channel; 2016 is measured on both halves together
    "qcd": E,  # QCD from ABCD, estimated per era
    "dy_table": E,
    "theory": K,  # cross sections and theory
}
# eras whose payload is a byte-identical copy of the 2018 one (E12): their nuisance is the 2018 one (K6)
PLACEHOLDER_ERAS = {
    "tau_vse": ("2016preVFP", "2016postVFP"),
    "tau_vsmu": ("2016preVFP", "2016postVFP", "2017"),
    "tau_e_es": ("2016preVFP", "2016postVFP"),
}
# the flat VSe + VSmu lnN is a copy where both components are: 2016; in 2017 only VSmu is, so it keeps the 2017 token
PLACEHOLDER_ERAS["tau_vslep"] = tuple(era for era in PLACEHOLDER_ERAS["tau_vse"] if era in PLACEHOLDER_ERAS["tau_vsmu"])
JES_YEAR_TOKEN = {"2016preVFP": "2016APV", "2016postVFP": "2016", "2017": "2017", "2018": "2018"}  # JME payload names


def era_token(family: str, era: str) -> str | None:
    """The era token of a nuisance of the family in the era; None for a nuisance correlated over all eras."""
    if era not in ERAS:
        raise ValueError(f"unknown era {era!r}")
    scheme = FAMILIES[family]
    if scheme == K:
        return None
    if scheme == J:
        return YEAR[era]
    if family == "jes_year":
        return JES_YEAR_TOKEN[era]
    return "2018" if era in PLACEHOLDER_ERAS.get(family, ()) else era


def nuisance(name: str, family: str, era: str) -> str:
    """The datacard name: `name` with the era token of the family appended."""
    token = era_token(family, era)
    return name if token is None else f"{name}_{token}"
