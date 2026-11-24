"""The names CROWN gives the shifts and the theory and scale-factor columns of the v8 shift production, in one place.

Source: sm_config of CROWN bbtautau built with -DSHIFTS=all (v8 spec section 4; the inventory per era, sample type
and scope is docs/analysen/v8/v8_spec_files/shift_inventory_after_paket1.json, its copy for the tests
tests/fixtures/v8_shift_inventory.json.gz, which tests/test_v8_shifts.py compares with what the analysis declares). A
shifted column is `<column>__<shift><direction>` (shifted()), the suffix of a shift `__<shift><direction>` (suffix()).
Every name carries the era of the n-tuples it is in: for a sample borrowed from another era that era, not the run's.

Not in the inventory, from the CROWN package that follows it (C23, C24, C40; the names are fixed, the nominal column of
the parton-shower weight is PS_WEIGHT):
- the weight shifts ps_isr and ps_fsr (PARTON_SHOWER) on every simulated sample;
- the columns lhe_pdf_weight_<i>, i = 1..100 (pdf_weight), of the signal sample type hh2b2tau in the nominal tree;
- the columns id_wgt_tau_vsJet_<WP>_highpt[_up|_down]_<leg> (highpt_columns) of every hadronic tau leg of the simulated
  samples in the nominal tree.
"""
from __future__ import annotations

DIRECTIONS = ("Up", "Down")
TAU_ID = "DeepTau2018v2p5"
DECAY_MODES = (0, 1, 10, 11)
ETA_REGIONS = ("barrel", "endcap")
WHEELS = (1, 2, 3, 4, 5)  # the muon-chamber wheels of the vsMu SF
# the regrouped JES sources: correlated between the years, and per era
JES_CORRELATED = ("Absolute", "BBEC1", "EC2", "FlavorQCD", "HF", "RelativeBal")
JES_UNCORRELATED = ("Absolute", "BBEC1", "EC2", "HF", "RelativeSample")
# the JERC regions of the JER (C33): |eta| < 1.93, 1.93-2.5, 2.5-3 and 3-5 each below and above 50 GeV
JER_REGIONS = ("eta0to1p93", "eta1p93to2p5", "eta2p5to3_ptlt50", "eta2p5to3_ptgt50", "eta3to5_ptlt50", "eta3to5_ptgt50")
# the |eta| ranges of the muon energy scale per scope (C37): mt has no muon beyond 2.1 (E31)
MUON_ES_REGIONS = {"mt": ("etalt1p2", "eta1p2to2p1"), "em": ("etalt1p2", "eta1p2to2p1", "eta2p1to2p4")}
VSJET_STAT = (1, 2)  # the two statistical components of the DM-binned vsJet SF
QCD_SCALES = ("QCDscale_ren", "QCDscale_fac")  # shifts of LHE_SCALE_WEIGHT: muR and muF times 2 (Up) and 0.5 (Down)
PARTON_SHOWER = ("ps_isr", "ps_fsr")  # shifts of PS_WEIGHT: the ISR and FSR scale times 2 (Up) and 0.5 (Down)
LHE_SCALE_WEIGHT = "lhe_scale_weight"
# ASSUMPTION: the output column of the CROWN parton-shower producer (C23), whose shifts are ps_isr and ps_fsr; the
# running CROWN package fixes the shift names, this column name is the one ShapeSmith expected before (SH14)
PS_WEIGHT = "ps_weight"
PDF_MEMBERS = tuple(range(1, 101))  # the Hessian eigenvectors of LHEPdfWeight (member 0 is the central one)
HIGHPT_THRESHOLD = 140.0  # GeV: from here the POG gives the pT-binned vsJet SF (C40)


def shifted(column: str, shift: str, direction: str) -> str:
    """The branch of `column` under the shift in a direction (Up or Down)."""
    return f"{column}__{shift}{direction}"


def suffix(shift: str, direction: str) -> str:
    """What a shift in a direction appends to every column it moves."""
    return f"__{shift}{direction}"


def jes(source: str, era: str | None = None) -> str:
    """A regrouped JES source: correlated between the years (no era), or the era's own."""
    return f"CMS_scale_j_{source}" if era is None else f"CMS_scale_j_{source}_{era}"


def jer(region: str, era: str) -> str:
    return f"CMS_res_j_{region}_{era}"


def unclustered_energy(era: str) -> str:
    """Not on the recoil-corrected samples (DY and W of 2018, C38)."""
    return f"CMS_scale_met_unclustered_energy_{era}"


def recoil(kind: str, era: str) -> str:
    """The recoil calibration, `kind` scale or res; only on the recoil-corrected samples."""
    return f"CMS_{kind}_met_RecoilCalibration_{era}"


def electron_es(kind: str, era: str) -> str:
    """The electron energy scale (`kind` scale) and resolution (res) of the simulation."""
    return f"CMS_{kind}_e_{era}"


def embedding_electron_es(region: str, era: str) -> str:
    """The electron energy scale of the et embedding, barrel and endcap (C32)."""
    return f"CMS_scale_e_{region}_emb_{era}"


def muon_es(region: str, era: str) -> str:
    """The muon energy scale of the simulation and the embedding (C37), `region` of MUON_ES_REGIONS."""
    return f"CMS_scale_m_{region}_{era}"


def tau_es(dm: int, era: str) -> str:
    """The energy scale of genuine hadronic taus per decay mode (C21)."""
    return f"CMS_scale_t_{TAU_ID}_DM{dm}_genTau_{era}"


def electron_fake_es(dm: int, region: str, era: str) -> str:
    """The energy scale of electrons reconstructed as hadronic taus; in et only (C29)."""
    return f"CMS_scale_t_{TAU_ID}_DM{dm}_genElectron_{region}_{era}"


def muon_fake_es(dm: int, era: str) -> str:
    """The energy scale of muons reconstructed as hadronic taus; in mt only (C21, C29)."""
    return f"CMS_scale_t_{TAU_ID}_DM{dm}_genMuon_{era}"


def embedding_tau_es(dm: int, era: str) -> str:
    return f"CMS_scale_t_emb_{TAU_ID}_DM{dm}_{era}"


def embedding_tau_id(dm: int, era: str) -> str:
    return f"CMS_eff_t_emb_{TAU_ID}_VSjet_DM{dm}_{era}"


def vsjet_stat(component: int, dm: int, era: str) -> str:
    return f"CMS_eff_t_{TAU_ID}_VSjet_dm_stat{component}_DM{dm}_{era}"


def vsjet_syst_alleras() -> str:
    return f"CMS_eff_t_{TAU_ID}_VSjet_dm_syst_alleras"


def vsjet_syst(era: str) -> str:
    return f"CMS_eff_t_{TAU_ID}_VSjet_dm_syst_{era}"


def vsjet_syst_tes(dm: int, era: str) -> str:
    """The vsJet SF component from the tau energy scale of its measurement (C18)."""
    return f"CMS_eff_t_{TAU_ID}_VSjet_dm_syst_TES_{era}_dm{dm}"


def vsele(dm: int, region: str, era: str) -> str:
    return f"CMS_fake_t_{TAU_ID}_VSe_DM{dm}_{region}_{era}"


def vsmu(wheel: int, era: str) -> str:
    return f"CMS_fake_t_{TAU_ID}_VSmu_wheel{wheel}_{era}"


def pileup(era: str) -> str:
    return f"CMS_pileup_{era}"


def lepton_sf(channel: str, era: str) -> tuple[str, ...]:
    """The ID, isolation and trigger SF shifts of the light leptons of a scope (em triggers on the muon and has no
    electron trigger shift; tt has no light lepton)."""
    muon = (f"CMS_eff_m_id_{era}", f"CMS_eff_m_iso_{era}", f"CMS_eff_m_trigger_{era}")
    return {"mt": muon, "et": (f"CMS_eff_e_id_{era}", f"CMS_eff_e_trigger_{era}"), "em": (*muon, f"CMS_eff_e_id_{era}")}.get(channel, ())


def ditau_trigger(dm: int, era: str) -> str:
    """The di-tau trigger SF per decay mode (C39); tt only."""
    return f"CMS_eff_trigger_tt_dm{dm}_{era}"


def pdf_weight(member: int) -> str:
    """The weight of a Hessian eigenvector of the PDF set (C24): a column of the nominal tree, signal only."""
    return f"lhe_pdf_weight_{member}"


def highpt_columns(wp: str, leg: int) -> dict[str, str]:
    """The pT-binned vsJet SF of a hadronic tau leg with its total uncertainty (C40): columns of the nominal tree, MC
    only; the nominal vsJet SF id_wgt_tau_vsJet_<WP>_<leg> stays the DM-binned one."""
    return {"nominal": f"id_wgt_tau_vsJet_{wp}_highpt_{leg}", "up": f"id_wgt_tau_vsJet_{wp}_highpt_up_{leg}", "down": f"id_wgt_tau_vsJet_{wp}_highpt_down_{leg}"}
