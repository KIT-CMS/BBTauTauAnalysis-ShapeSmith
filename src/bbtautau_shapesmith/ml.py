"""The ML export of the tau channels (NN spec 3.1, X1-X7): label profiles, classes and variables.

Label profiles (switch ml_labels; the class order is the class index of the export and of the NN categories):
- thesis6 (jet_fakes: mc): the six classes of the master's thesis, HH, DY = ZTT + ZL + ZJ (with embedding EMB), ST,
  TT, VV and Other (W, TTV, EWK, single Higgs); the QCD of the ABCD method is not exported;
- v7b5 (jet_fakes: ff, N3, spec 6.1): HH; tautau = every genuine di-tau process (EMB in the embedded channels, else
  ZTT, TTT, STT, VVT, TTVT and EWKT); Top = TTL + STL; jetFakes = the data of the anti-isolated region times the fake
  factor, with the subtracted processes as subtraction rows; Other = ZL, VVL, TTVL, EWKL and single Higgs.
Variables (switch ml_variables): column names or the names of the sets below; default every set. reskim and truth are
not in the smrun2_binned_v7b skims before the re-skim of X8 (in the n-tuples they are), truth exists for simulated
samples only (NaN in the data and embedded rows). mtt_coll_approx of the thesis does not exist in v15; the model
computes it from the legs and the MET. iso_1 is the light-lepton isolation in et and mt, but the raw DeepTau vsJet
score of the leading tau in tt, which defines the anti-isolated region: no NN input there (spec 5.3).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from shapesmith.model import AnalysisError, MLExportConfig, Process

from bbtautau_shapesmith.processes import EMBEDDED, GENUINE, SPLITS

if TYPE_CHECKING:
    from bbtautau_shapesmith.switches import TauSwitches

ML_PROFILES = {"thesis6": "mc", "v7b5": "ff"}  # label profile -> the jet_fakes switch it needs
ML_CLASSES = {
    "thesis6": ("HH2B2Tau", "DY", "ST", "TT", "VV", "Other"),  # the order of the thesis (config.py: Signal, DY, ...)
    "v7b5": ("HH2B2Tau", "tautau", "Top", "jetFakes", "Other"),
}
THESIS_CLASS_OF_GROUP = {"DY": "DY", EMBEDDED: "DY", "ST": "ST", "TT": "TT", "VV": "VV"}
TOP = frozenset({SPLITS["TT"]["L"], SPLITS["ST"]["L"]})
ML_IDENTIFIERS = ("run", "lumi", "event")
# the 20 inputs of the thesis (columns_expanded) with the v15 names (nbtag -> n_bjets, njets -> n_jets), without
# mtt_coll_approx
THESIS_INPUTS = (
    "bpair_btag_value_1", "bpair_btag_value_2", "bpair_deltaR", "bpair_m_inv", "bpair_pt_1", "bpair_pt_dijet",
    "deltaR_ditaupair", "jpt_1", "m_fastmtt", "mjj", "n_bjets", "n_jets", "pt_fastmtt", "m_vis", "mt_tot", "jpt_2",
    "pt_1", "pt_dijet", "pt_vis",
)
# the object slots of spec 5.3 that the v7b skims hold: both legs, the b pair, the two leading jets, MET, FastMTT
OBJECTS = (
    "pt_1", "eta_1", "phi_1", "mass_1", "iso_1", "tau_decaymode_1",
    "pt_2", "eta_2", "phi_2", "mass_2", "tau_decaymode_2",
    "bpair_pt_1", "bpair_eta_1", "bpair_phi_1", "bpair_btag_value_1", "bpair_pt_2", "bpair_eta_2", "bpair_phi_2",
    "bpair_btag_value_2", "jpt_1", "jeta_1", "jphi_1", "jpt_2", "jeta_2", "jphi_2", "met", "metphi", "metSumEt",
    "pt_fastmtt", "eta_fastmtt", "phi_fastmtt", "m_fastmtt",
)
KINEMATICS = (
    "mt_1", "mt_2", "mt_tot", "pzetamissvis", "mass_tautaubb", "pt_tautaubb", "deltaR_ditaupair", "bpair_deltaR",
)
RESKIM = ("bpair_mass_1", "bpair_mass_2", "jtag_value_1", "jtag_value_2")  # X8: in the n-tuples, not in the skims
TRUTH = ("bpair_hadronFlavour_1", "bpair_hadronFlavour_2", "jet_hadronFlavour_1", "jet_hadronFlavour_2")  # X8, MC
VARIABLE_SETS = {
    "thesis": THESIS_INPUTS, "objects": OBJECTS, "kinematics": KINEMATICS, "reskim": RESKIM, "truth": TRUTH,
}
# the generator-truth columns of the simulated samples only (the GenJet flavour as in processes.HADRON_FLAVOURS)
MC_ONLY = frozenset(TRUTH) | {"genjet_hadFlavour_1", "genjet_hadFlavour_2"}
FASTMTT_COLUMNS = frozenset({"m_fastmtt", "pt_fastmtt", "eta_fastmtt", "phi_fastmtt"})  # from the FastMTT friend
# the groups of the raw-count rule of the data-blind NN binning (spec 6.3: >= 1 unweighted event of Top, tautau and
# jetFakes per bin), by the jet-fake mode of the background model: with fake factors the v7b5 classes Top (TTL + STL)
# and tautau (EMB or the genuine MC processes) and the jetFakes estimate (its anti-isolated data); in mc mode the
# corresponding major backgrounds of the thesis classes, TT (TTT + TTL + TTJ) and DY (ZTT or EMB, ZL, ZJ), and the QCD of
# the ABCD method (its data of the anti-isolated region)
RAW_COUNT_CLASSES = {"ff": ("v7b5", ("Top", "tautau"), "jetFakes"), "mc": ("thesis6", ("TT", "DY"), "QCD")}


def variables(names: list[str] | None) -> tuple[str, ...]:
    """The export variables of the switch ml_variables: the sets expanded, every column once, in order."""
    chosen = names if names is not None else VARIABLE_SETS
    return tuple(dict.fromkeys(column for name in chosen for column in VARIABLE_SETS.get(name, (name,))))


def class_of(process: Process, profile: str) -> str:
    """The class of a signal or background process in a label profile (the jet fakes estimate is added apart)."""
    if process.role == "signal":
        return "HH2B2Tau"
    if profile == "thesis6":
        return THESIS_CLASS_OF_GROUP.get(process.group, "Other")
    if process.name in GENUINE:
        return "tautau"
    return "Top" if process.name in TOP else "Other"


def ml_config(processes: tuple[Process, ...], switches: TauSwitches) -> MLExportConfig | None:
    """The ML export of the label profile of the switch ml_labels (None: no export); in fake-factor mode with the
    jetFakes estimate, the data of the anti-isolated region, and its subtraction rows."""
    profile = switches.ml_labels
    if profile is None:
        return None
    columns = variables(switches.ml_variables)
    fastmtt = sorted(FASTMTT_COLUMNS & set(columns))
    if fastmtt and not switches.fastmtt_friend:
        raise AnalysisError(f"ml_variables {fastmtt} need the FastMTT friend (fastmtt_friend: true)")
    classes = {p.name: class_of(p, profile) for p in processes if p.role in ("signal", "background")}
    region_of = {}
    if switches.jet_fakes == "ff":
        classes["jetFakes"] = "jetFakes"
        region_of["jetFakes"] = "anti_iso"
    return MLExportConfig(
        variables=columns, classes=ML_CLASSES[profile], class_of=classes, region_of=region_of,
        identifiers=ML_IDENTIFIERS, mc_only=tuple(v for v in columns if v in MC_ONLY), profile=profile,
    )


def raw_count_groups(processes: tuple[Process, ...], jet_fakes: str) -> dict[str, tuple[str, ...]]:
    """The raw-count groups (RAW_COUNT_CLASSES) of a channel's process table: label -> its background processes; a
    class without one is left out."""
    profile, labels, output = RAW_COUNT_CLASSES[jet_fakes]
    backgrounds = [p for p in processes if p.role == "background"]
    groups = {label: tuple(p.name for p in backgrounds if class_of(p, profile) == label) for label in labels}
    return {**{label: names for label, names in groups.items() if names}, output: (output,)}
