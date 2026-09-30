"""The analysis switches of the run YAML (`switches:`), typed and checked; an unknown or mistyped switch raises."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from shapesmith.model import AnalysisError

STRICT = ConfigDict(extra="forbid", strict=True, frozen=True)
SampleLists = list[str]  # inventory/<name>.txt: the KingMaker sample lists of the production


class TauSwitches(BaseModel):
    model_config = STRICT

    sample_lists: SampleLists = Field(min_length=1)
    jet_fakes: Literal["mc", "ff"] = "mc"  # MC jet fakes + ABCD QCD, or the fake-factor estimate (FF friend)
    embedding: bool = False  # genuine tautau from the embedded samples instead of MC
    nn_friend: bool = False  # NN categories from the NN friend
    control_regions: bool = False  # wider skim plus the named pass/fail control regions (needs jet_fakes: mc)
    shape_systematics: bool = True  # the embedding and fake-factor shifts (column variations)

    @model_validator(mode="after")
    def _controls_need_mc_fakes(self) -> TauSwitches:
        if self.control_regions and self.jet_fakes != "mc":
            raise ValueError("control_regions requires jet_fakes: mc for raw pass/fail data/MC comparisons")
        return self


class DileptonSwitches(BaseModel):
    model_config = STRICT

    sample_lists: SampleLists = Field(min_length=1)


def parse(model: type[BaseModel], switches: dict) -> BaseModel:
    try:
        return model.model_validate(switches)
    except ValidationError as error:
        raise AnalysisError(f"invalid switches: {error}") from error
