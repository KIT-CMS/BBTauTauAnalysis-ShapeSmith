"""The analysis switches of the run YAML (`switches:`), typed and checked; an unknown or mistyped switch raises."""
from __future__ import annotations

import logging
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator
from shapesmith.model import AnalysisError

from bbtautau_shapesmith.constants import RECOIL_ERAS
from bbtautau_shapesmith.ml import ML_PROFILES

STRICT = ConfigDict(extra="forbid", strict=True, frozen=True)
SampleLists = list[str]  # inventory/<name>.txt: the KingMaker sample lists of the production

logger = logging.getLogger(__name__)


def _distinct(embedding: bool | list[str]) -> bool | list[str]:
    if isinstance(embedding, list) and len(set(embedding)) < len(embedding):
        raise ValueError(f"a channel is listed twice: {embedding}")
    return embedding


# true or false for every tau channel, or the embedded channels (a list: strict pydantic takes no YAML list as tuple)
Embedding = Annotated[bool | list[Literal["et", "mt", "tt"]], AfterValidator(_distinct)]
# njets_btag: every DY process times the data/MC table DY_NJETS_BTAG (constants.py) of its n_bjets and n_jets
DyCorrection = Literal["none", "njets_btag"]
# the processes with own fake factors (v7 spec E18, 6.8): QCD and ttbar (the v6 method), or QCD, W+jets and ttbar
FF_VARIANTS = {("QCD", "ttbar"): "fake_factors", ("QCD", "Wjets", "ttbar"): "fake_factors_w"}  # -> measurement name


def _ff_variant(processes: list[str]) -> list[str]:
    if tuple(processes) not in FF_VARIANTS:
        raise ValueError(f"ff_processes must be one of {[list(v) for v in FF_VARIANTS]}, got {processes}")
    return processes


FfProcesses = Annotated[list[Literal["QCD", "Wjets", "ttbar"]], AfterValidator(_ff_variant)]


class _FakeFactorProcesses:
    """The methods of the switch `ff_processes`."""

    @property
    def wjets(self) -> bool:
        """W+jets fake factors (the variant QCD + W + ttbar)."""
        return "Wjets" in self.ff_processes


class _EmbeddedChannels:
    """The channel methods of the switch `embedding`."""

    def embedded(self, channel: str) -> bool:
        """Genuine taus of the channel from the embedded samples."""
        return channel in self.embedding if isinstance(self.embedding, list) else self.embedding

    def check_embedding(self, channels: list[str]) -> None:
        """A listed embedded channel must be a channel of the run."""
        unknown = [channel for channel in self.embedding if channel not in channels] if isinstance(self.embedding, list) else []
        if unknown:
            raise AnalysisError(f"embedding: {', '.join(unknown)} not among the channels {', '.join(channels)}")


class TauSwitches(_EmbeddedChannels, _FakeFactorProcesses, BaseModel):
    model_config = STRICT

    sample_lists: SampleLists = Field(min_length=1)
    jet_fakes: Literal["mc", "ff"] = "mc"  # MC jet fakes + ABCD QCD, or the fake-factor estimate (FF friend)
    ff_type: Literal["corrected", "raw"] = "corrected"  # FF friend weight; raw: no DR->SR and non-closure corrections (needs jet_fakes: ff, shape_systematics: false)
    embedding: Embedding = False  # genuine tautau from the embedded samples instead of MC, in every tau channel or the listed ones
    nn_friend: bool = False  # NN categories from the NN friend
    # the model of the NN friend (SH1): the column prefix of the model (one friend can hold several models; columns
    # <prefix>predicted_class, <prefix>predicted_max_value, <prefix>nn_score_<class>, -1 outside its gate) and its
    # classes in the order of predicted_class (none: those of the label profile, ml_labels or else thesis6 with
    # jet_fakes mc and v7b5 with ff); nn_manifest: the manifest.json of the model's CROWN payload ({channel} replaced by
    # every tau channel of the run), whose "classes" must be exactly these classes
    nn_prefix: Annotated[str, Field(pattern=r"^[A-Za-z0-9_]*$")] = ""
    nn_classes: list[str] | None = None
    nn_manifest: str | None = None
    # the column prefixes of every model of the NN friend production (the MODELS of its CROWN configuration): the skims
    # keep <prefix>predicted_class and <prefix>predicted_max_value of each, so that one skim serves the run of every
    # model (nn_prefix picks it, -s switches.nn_prefix=...); empty: the columns of nn_prefix only
    nn_friend_prefixes: list[Annotated[str, Field(pattern=r"^[A-Za-z0-9_]*$")]] = Field(default_factory=list)
    # the NN categories (variables.py): thesis (argmax, R3-R5), k1 (argmax without n_bjets >= 3, N12) or k2 (k1 with the
    # HH category split by n_bjets == 1 / == 2, N10); their binning, resolved by datacards: data_blind (NN spec 6.3) or
    # thesis (the custom edges and AutoRebin(1.0, 0.9) of the thesis)
    nn_categories: Literal["thesis", "k1", "k2"] = "k1"
    nn_binning: Literal["data_blind", "thesis"] = "data_blind"
    # the ML export (ml.py, NN spec 3.1): the label profile thesis6 (needs jet_fakes: mc) or v7b5 (needs jet_fakes: ff),
    # none without an export; the class order of the profile is also that of the NN categories
    ml_labels: Literal["thesis6", "v7b5"] | None = None
    # the export variables: column names or the variable sets of ml.py (thesis, objects, kinematics, reskim, truth);
    # none: every set (reskim and truth need the re-skim of X8)
    ml_variables: list[str] | None = None
    # the signal process HH2B2Tau from the powheg kappa_lambda = 1 sample (E23), or from the madgraph node_SM sample of
    # the master's thesis (R3, N9; the powheg sample is then left out)
    signal_sample: Literal["powheg", "node_sm"] = "powheg"
    # the control variables m, pt, eta and phi_fastmtt of the di-tau system from the FastMTT friend (CROWN sm_fastmtt,
    # tag fastmtt; et, mt and tt, em has none)
    fastmtt_friend: bool = False
    # nn: the NN categories (needs nn_friend); control: one category 'nominal' per channel; none: no fit categories
    # (a run of control plots)
    fit_categories: Literal["nn", "control", "none"] = "nn"
    control_regions: bool = False  # wider skim plus the named pass/fail control regions (needs jet_fakes: mc)
    diagnostics: bool = False  # QCD in the control regions and the diagnostic variables (needs control_regions; npv_good, rho from v6)
    shape_systematics: bool = True  # the embedding and fake-factor shifts (column variations)
    # the production has CROWN shifts (the embedding tau ES and vsJet ID); false for a nominal-only production (v7, E34),
    # which keeps the fake-factor friend shifts and the weight variations of the nominal tree
    crown_shifts: bool = True
    dy_correction: DyCorrection = "none"  # on every DY process (ZTT, ZL, ZJ)
    # the fake-factor payload variant of the friend: the CROWN shifts of its W+jets keys in et and mt (jet_fakes: ff)
    ff_processes: FfProcesses = Field(default_factory=lambda: ["QCD", "ttbar"])
    # the uncertainties sized in the v7b studies (study_uncertainties.py, systematics_v7b_summary.md): embedding tau ID
    # (weight factors) and ES, POG MC tau ES, vsJet components (in place of the flat eff_t_vsJet) and di-tau trigger SF,
    # MET residual, HIPM, the fake-factor decay-mode non-closure (shape) and 2016 halves, K4; only for the fake-factor run
    # of a nominal-only production (the MC tau ES and the vsJet replacement of the MC jet fakes have no other source)
    study_uncertainties: bool = False
    # the datacards symmetrise the fake-factor friend shifts whose Up and Down move a process to the same side of the
    # nominal or by more than twice as much on one side (Analysis.symmetrised_variations, shapesmith.datacards; on in the
    # v7b fake-factor runs since 2026-10-19: the mt QCD non-closure shifts of 2017/2018 at -8/+28 % made robustHesse drop
    # them from the covariance); needs jet_fakes: ff
    symmetrise_ff_shifts: bool = False
    # the systematics stages of the NN phase (nn_limit_spec.md 8, systematics.py): S0+ the top-pT shape of ttbar
    # CMS_htt_ttbarShape (SH7) and the 30 % normalisation of the heavy-flavour component at >= 3 b tags
    # CMS_htt_heavy_flavour (SH8; needs skims with bpair_hadronFlavour_{1,2}, as the re-skim of NN spec X8), on the
    # tau channels and em; S1 the signal theory shapes normalised to the inclusive rate (SH14): qcd_scale (the CROWN
    # shifts QCDscale_ren and QCDscale_fac of lhe_scale_weight), parton_shower and pdf (columns the shift production P6 is
    # to add); they need crown_shifts: true and the powheg signal
    ttbar_shape: bool = False
    heavy_flavour: bool = False
    signal_theory: list[Literal["qcd_scale", "parton_shower", "pdf"]] = Field(default_factory=list)
    # the input validation of the NN phase (nn_validation.py, nn_limit_spec.md 7.1, NV1-NV3): the control variables are
    # the NN inputs of each channel in fine bins (shapesmith input-gof merges them), the eta-phi occupancy of the legs and
    # jets and yield; the skims keep the inputs of FastMTT with the MET covariance (a refit, scripts/fastmtt_refit.py)
    nn_validation: bool = False
    # PRELIMINARY pseudo-shifts of the jet (3 %) and tau_h (1.5 %) energies with the MET propagated, as shape nuisances of
    # the jet, MET and FastMTT inputs until P6 (NV1 with NV9 size, nn_validation.py), symmetrised in the datacards; needs
    # nn_validation
    pseudo_shifts: bool = False
    # the anti-isolated region anti_iso with the fake-factor weight of the friend and its shifts in mc mode: the shapes
    # of the MC jet fakes there against the nominal region (NV1); needs jet_fakes: mc and the fake-factor friend
    ff_transfer: bool = False

    @model_validator(mode="after")
    def _nn_categories_need_the_nn_friend(self) -> TauSwitches:
        # without the friend there would be no tau category at all, and datacards with em only (NN spec X9)
        if self.fit_categories == "nn" and not self.nn_friend:
            raise ValueError("fit_categories nn requires nn_friend: true (without the NN friend: control or none)")
        return self

    @model_validator(mode="after")
    def _nn_classes_hold_the_signal_once(self) -> TauSwitches:
        if self.nn_classes is not None and (self.nn_classes.count("HH2B2Tau") != 1 or len(set(self.nn_classes)) != len(self.nn_classes)):
            raise ValueError(f"nn_classes must be distinct and contain HH2B2Tau (the signal category), got {self.nn_classes}")
        return self

    @model_validator(mode="after")
    def _nn_prefix_is_a_model_of_the_friend(self) -> TauSwitches:
        prefixes = self.nn_friend_prefixes
        if prefixes and (self.nn_prefix not in prefixes or len(set(prefixes)) < len(prefixes)):
            raise ValueError(f"nn_friend_prefixes must be distinct and contain nn_prefix '{self.nn_prefix}', got {prefixes}")
        return self

    @model_validator(mode="after")
    def _ml_labels_match_the_jet_fakes(self) -> TauSwitches:
        if self.ml_labels is not None and ML_PROFILES[self.ml_labels] != self.jet_fakes:
            raise ValueError(f"ml_labels {self.ml_labels} requires jet_fakes: {ML_PROFILES[self.ml_labels]}")
        if self.ml_variables is not None and self.ml_labels is None:
            raise ValueError("ml_variables requires ml_labels")
        return self

    @model_validator(mode="after")
    def _controls_need_mc_fakes(self) -> TauSwitches:
        if self.control_regions and self.jet_fakes != "mc":
            raise ValueError("control_regions requires jet_fakes: mc for raw pass/fail data/MC comparisons")
        return self

    @model_validator(mode="after")
    def _diagnostics_need_control_regions(self) -> TauSwitches:
        if self.diagnostics and not self.control_regions:
            raise ValueError("diagnostics requires control_regions: true")
        return self

    @model_validator(mode="after")
    def _study_uncertainties_need_a_nominal_fake_factor_run(self) -> TauSwitches:
        if self.study_uncertainties and (self.jet_fakes != "ff" or self.crown_shifts):
            raise ValueError("study_uncertainties requires jet_fakes: ff and crown_shifts: false")
        return self

    @model_validator(mode="after")
    def _symmetrised_shifts_need_fake_factors(self) -> TauSwitches:
        if self.symmetrise_ff_shifts and self.jet_fakes != "ff":
            raise ValueError("symmetrise_ff_shifts requires jet_fakes: ff")
        return self

    @model_validator(mode="after")
    def _signal_theory_needs_the_shift_production(self) -> TauSwitches:
        if len(set(self.signal_theory)) < len(self.signal_theory):
            raise ValueError(f"signal_theory lists a kind twice: {self.signal_theory}")
        if self.signal_theory and not self.crown_shifts:
            raise ValueError("signal_theory requires crown_shifts: true (the theory weights come with the shift production P6)")
        if self.signal_theory and self.signal_sample != "powheg":
            raise ValueError("signal_theory requires signal_sample: powheg (the normalisation is that of the powheg sample)")
        return self

    @model_validator(mode="after")
    def _validation_switches(self) -> TauSwitches:
        if self.pseudo_shifts and not self.nn_validation:
            raise ValueError("pseudo_shifts requires nn_validation: true (preliminary nuisances of the input validation only)")
        if self.ff_transfer and self.jet_fakes != "mc":
            raise ValueError("ff_transfer requires jet_fakes: mc (the MC jet fakes in the anti-isolated region)")
        return self

    @model_validator(mode="after")
    def _raw_fake_factors_are_nominal_only(self) -> TauSwitches:
        if self.ff_type == "raw" and self.jet_fakes != "ff":
            raise ValueError("ff_type raw requires jet_fakes: ff")
        if self.ff_type == "raw" and self.shape_systematics:
            raise ValueError("ff_type raw supports nominal control plots only; set shape_systematics: false")
        return self


class FakeFactorSwitches(_EmbeddedChannels, _FakeFactorProcesses, BaseModel):
    model_config = STRICT

    sample_lists: SampleLists = Field(min_length=1)
    embedding: Embedding = False  # genuine taus subtracted as EMB instead of MC (not in the DR->SR steps), in every or the listed channels
    dy_correction: DyCorrection = "none"  # also on the subtracted DY processes
    # the tt fraction region of a leg: both taus anti-isolated (U6, jvoss parity), or its application region (the leg
    # anti-isolated, the other tau isolated), where the fake factor is applied
    tt_fraction_region: Literal["both_anti_iso", "application"] = "both_anti_iso"
    # QCD + ttbar (output fake_factors/) or QCD + W + ttbar (fake_factors_w/; the W+jets fake factors in et and mt)
    ff_processes: FfProcesses = Field(default_factory=lambda: ["QCD", "ttbar"])


class DileptonSwitches(BaseModel):
    model_config = STRICT

    sample_lists: SampleLists = Field(min_length=1)
    dy_correction: DyCorrection = "none"  # none to derive the table, njets_btag for its closure
    diagnostics: bool = False  # n_jets regions, flavour templates of DY (mm, ee) and ttbar (em), the diagnostic variables
    # the input validation of the NN phase (NV2, NV3): >= 2 jets in every channel, the b-tag scores and jet variables of
    # nn_validation.py as control variables, the inputs of FastMTT with the MET covariance kept (a refit)
    nn_validation: bool = False


class TauIdSwitches(BaseModel):
    model_config = STRICT

    sample_lists: SampleLists = Field(min_length=1)
    vsjet_wp: Literal["Medium", "Tight"] = "Tight"  # the measured DeepTau vsJet working point
    vsele_wp: Literal["VVLoose", "Tight"] = "VVLoose"  # the DeepTau vsEle working point it is measured with
    shape_systematics: bool = True  # the MC shape uncertainties (CROWN shifts and weight variations)
    # the eras whose DY and W carry the recoil shifts in the production: constants.RECOIL_ERAS (2018, the MET fix,
    # smrun2_tau_id_v2 on); smrun2_tau_id_v1 has them in every era
    recoil_eras: list[Literal["2016preVFP", "2016postVFP", "2017", "2018"]] = Field(default_factory=lambda: list(RECOIL_ERAS))


class TauTriggerSwitches(BaseModel):
    """The tau-leg trigger measurement (tau_trigger_measurement.py, E33)."""

    model_config = STRICT

    sample_lists: SampleLists = Field(min_length=1)
    wp_class: Literal["Medium", "VVVLoose_not_Medium"] = "Medium"  # the isolation class of the probe tau
    # the selection: nominal, or a variation whose change is an uncertainty of the merged payload
    variant: Literal["nominal", "signal_mc", "tag_pt", "mt30", "monitor", "closure"] = "nominal"
    emb_es_payload: str | None = None  # the tau-ID/ES payload of the era (5.4): the energy scale of the embedded taus
    dm1011: bool = False  # decay modes 10 and 11 in one bin (the anti-isolated class, where single fits do not converge)
    reference_payload: str | None = None  # a payload with tau_trigger (the POG tau.json.gz) drawn in the plots
    kit_reference_payload: str | None = None  # the KIT embedding file (tauTriggerSF: eff_data, eff_emb) drawn in the plots


def parse(model: type[BaseModel], switches: dict) -> BaseModel:
    try:
        parsed = model.model_validate(switches)
    except ValidationError as error:
        raise AnalysisError(f"invalid switches: {error}") from error
    logger.info(f"switches ({model.__name__}, defaults included): {parsed.model_dump()}")
    return parsed
