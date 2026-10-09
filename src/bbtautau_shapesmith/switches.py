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


class TauSwitches(_EmbeddedChannels, BaseModel):
    model_config = STRICT

    sample_lists: SampleLists = Field(min_length=1)
    jet_fakes: Literal["mc", "ff"] = "mc"  # MC jet fakes + ABCD QCD, or the fake-factor estimate (FF friend)
    ff_type: Literal["corrected", "raw"] = "corrected"  # FF friend weight; raw: no DR->SR and non-closure corrections (needs jet_fakes: ff, shape_systematics: false)
    embedding: Embedding = False  # genuine tautau from the embedded samples instead of MC, in every tau channel or the listed ones
    nn_friend: bool = False  # NN categories from the NN friend
    control_regions: bool = False  # wider skim plus the named pass/fail control regions (needs jet_fakes: mc)
    shape_systematics: bool = True  # the embedding and fake-factor shifts (column variations)
    dy_correction: DyCorrection = "none"  # on every DY process (ZTT, ZL, ZJ)

    @model_validator(mode="after")
    def _controls_need_mc_fakes(self) -> TauSwitches:
        if self.control_regions and self.jet_fakes != "mc":
            raise ValueError("control_regions requires jet_fakes: mc for raw pass/fail data/MC comparisons")
        return self

    @model_validator(mode="after")
    def _raw_fake_factors_are_nominal_only(self) -> TauSwitches:
        if self.ff_type == "raw" and self.jet_fakes != "ff":
            raise ValueError("ff_type raw requires jet_fakes: ff")
        if self.ff_type == "raw" and self.shape_systematics:
            raise ValueError("ff_type raw supports nominal control plots only; set shape_systematics: false")
        return self


class FakeFactorSwitches(_EmbeddedChannels, BaseModel):
    model_config = STRICT

    sample_lists: SampleLists = Field(min_length=1)
    embedding: Embedding = False  # genuine taus subtracted as EMB instead of MC (not in the DR->SR steps), in every or the listed channels
    dy_correction: DyCorrection = "none"  # also on the subtracted DY processes


class DileptonSwitches(BaseModel):
    model_config = STRICT

    sample_lists: SampleLists = Field(min_length=1)
    dy_correction: DyCorrection = "none"  # none to derive the table, njets_btag for its closure


class TauIdSwitches(BaseModel):
    model_config = STRICT

    sample_lists: SampleLists = Field(min_length=1)
    vsjet_wp: Literal["Medium", "Tight"] = "Tight"  # the measured DeepTau vsJet working point
    vsele_wp: Literal["VVLoose", "Tight"] = "VVLoose"  # the DeepTau vsEle working point it is measured with
    shape_systematics: bool = True  # the MC shape uncertainties (CROWN shifts and weight variations)


def parse(model: type[BaseModel], switches: dict) -> BaseModel:
    try:
        parsed = model.model_validate(switches)
    except ValidationError as error:
        raise AnalysisError(f"invalid switches: {error}") from error
    logger.info(f"switches ({model.__name__}, defaults included): {parsed.model_dump()}")
    return parsed
