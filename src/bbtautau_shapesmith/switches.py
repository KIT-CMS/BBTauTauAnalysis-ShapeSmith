"""The analysis switches of the run YAML (`switches:`), typed and checked; an unknown or mistyped switch raises."""
from __future__ import annotations

import logging
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator
from shapesmith.model import AnalysisError

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
    fit_categories: Literal["nn", "control"] = "nn"  # nn: the NN categories (with nn_friend); control: one category 'nominal' per channel
    control_regions: bool = False  # wider skim plus the named pass/fail control regions (needs jet_fakes: mc)
    diagnostics: bool = False  # QCD in the control regions and the diagnostic variables (needs control_regions; npv_good, rho from v6)
    shape_systematics: bool = True  # the embedding and fake-factor shifts (column variations)
    # the production has CROWN shifts (the embedding tau ES and vsJet ID); false for a nominal-only production (v7, E34),
    # which keeps the fake-factor friend shifts and the weight variations of the nominal tree
    crown_shifts: bool = True
    dy_correction: DyCorrection = "none"  # on every DY process (ZTT, ZL, ZJ)
    # the fake-factor payload variant of the friend: the CROWN shifts of its W+jets keys in et and mt (jet_fakes: ff)
    ff_processes: FfProcesses = Field(default_factory=lambda: ["QCD", "ttbar"])

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


class TauIdSwitches(BaseModel):
    model_config = STRICT

    sample_lists: SampleLists = Field(min_length=1)
    vsjet_wp: Literal["Medium", "Tight"] = "Tight"  # the measured DeepTau vsJet working point
    vsele_wp: Literal["VVLoose", "Tight"] = "VVLoose"  # the DeepTau vsEle working point it is measured with
    shape_systematics: bool = True  # the MC shape uncertainties (CROWN shifts and weight variations)


class TauTriggerSwitches(BaseModel):
    """The tau-leg trigger measurement (tau_trigger_measurement.py, E33)."""

    model_config = STRICT

    sample_lists: SampleLists = Field(min_length=1)
    wp_class: Literal["Medium", "VVVLoose_not_Medium"] = "Medium"  # the isolation class of the probe tau
    # the selection: nominal, or a variation whose change is an uncertainty of the merged payload
    variant: Literal["nominal", "signal_mc", "tag_pt", "mt30", "monitor"] = "nominal"
    emb_es_payload: str | None = None  # the tau-ID/ES payload of the era (5.4): the energy scale of the embedded taus
    dm1011: bool = False  # decay modes 10 and 11 in one bin (the anti-isolated class, where single fits do not converge)
    reference_payload: str | None = None  # a payload with tau_trigger (the POG tau.json.gz) drawn in the plots


def parse(model: type[BaseModel], switches: dict) -> BaseModel:
    try:
        parsed = model.model_validate(switches)
    except ValidationError as error:
        raise AnalysisError(f"invalid switches: {error}") from error
    logger.info(f"switches ({model.__name__}, defaults included): {parsed.model_dump()}")
    return parsed
