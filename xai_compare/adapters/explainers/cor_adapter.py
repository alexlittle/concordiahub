import json

import pandas as pd
from typing import Any
from dataclasses import dataclass

from xai_compare.context import RunContext, ResponseSchema
from xai_compare.exceptions import MissingDataError
from xai_compare.registry.xai_registry import register_xai
from xai_compare.adapters.explainers.xaimethod_base import XAIMethodAdapter
from xai_compare import strings, config


@dataclass
class ContribScore:
    score: int
    reasoning: str

@dataclass
class LLMResponse:
    reasoning: str
    prediction: int | str
    confidence: float
    contributions: dict[str, ContribScore]
    raw: dict

    def to_dict(self, sample_id_col: str, sample_id: str) -> dict:
        return {
            sample_id_col: sample_id,
            "reasoning": self.reasoning,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "contributions": {
                feature: {"score": c.score, "reasoning": c.reasoning}
                for feature, c in self.contributions.items()
            }
        }

@register_xai("cor")
class CoRAdapter(XAIMethodAdapter):
    """For chain-of-reasoning/chain-of-thought explanations"""

    display_name = strings.COR_DISPLAY_NAME
    default_visualisation = config.COR_DEFAULT_VIZ
    compatible_visualisations = config.COR_COMPATIBLE_VIZ

    # no features as such so will process the explanations in this adapter
    requires_encoded_features: bool = False

    # uses a json file for explanation storage
    explanation_file_type: str = "json"

    def explain_local_df(self, ctx: RunContext, model=None) -> pd.DataFrame:
        raise NotImplementedError

    def explain_local_json(self, ctx: RunContext) -> list[dict[str, Any]]:
        """Processes the CoR local explanations
        Args:
            ctx: RunContext object

        Returns:
            list[dict[str, Any]]: list of dicts containing local explanations

        Raises:
            MissingDataError: if data.x is not provided
        """
        if ctx.data.y_exp is None:
            raise MissingDataError("CoRAdapter requires y_exp in RunData")
        if ctx.response_schema is None:
            raise MissingDataError(
                "CoRAdapter requires a ResponseSchema on RunContext")

        sample_id_col = ctx.data.data_schema.sample_id_column
        output = []
        for idx, row in ctx.data.y_exp.iterrows():
            raw = json.loads(row["explanation"])
            response = self._normalise(raw, ctx.response_schema)
            output.append(response.to_dict(sample_id_col, row[sample_id_col]))
        return output

    def _normalise(self, raw: dict, schema: ResponseSchema) -> LLMResponse:
        raw_contribs = raw[schema.contributions_field]

        if schema.contribution_key is not None:
            contributions = {
                item[schema.contribution_key]: ContribScore(
                    score=item["score"],
                    reasoning=item["reasoning"],
                )
                for item in raw_contribs
            }
        else:
            contributions = {
                feature: ContribScore(
                    score=values["score"],
                    reasoning=values["reasoning"],
                )
                for feature, values in raw_contribs.items()
            }

        return LLMResponse(
            reasoning=raw["reasoning"],
            prediction=raw["prediction"],
            confidence=raw["confidence"],
            contributions=contributions,
            raw=raw,
        )