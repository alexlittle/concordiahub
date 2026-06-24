import json

import pandas as pd
from typing import Any
from dataclasses import dataclass

from xailens.entities.context import ModelContext, ResponseSchema
from xailens.exceptions import MissingDataError
from xailens.registry.xai_registry import register_xai
from xailens.adapters.explainers.xaimethod_base import XAIMethodAdapter
from xailens import strings, config, constants
from xailens.entities.modeleval import ModelEval


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

    def to_dict(self, instance_id_col: str, instance_id: str) -> dict:
        return {
            instance_id_col: instance_id,
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
    display_notes = strings.COR_NOTES
    default_visualisation = config.COR_DEFAULT_VIZ
    compatible_visualisations = config.COR_COMPATIBLE_VIZ

    # no features as such so will process the explanations in this adapter
    requires_encoded_features: bool = False

    # uses a json file for explanation storage
    explanation_file_type: str = "json"

    @classmethod
    def load_explanation(cls, modeleval: ModelEval, instance_id: str) -> pd.DataFrame:
        records = modeleval.store.load_json(f"exp_local_{cls.key}")
        instance_id_col = modeleval.get(constants.DATA_SCHEMA_ID_COLUMN)

        row = next((r for r in records if r[instance_id_col] == instance_id), None)
        if row is None:
            raise ValueError(
                f"Instance '{instance_id}' not found in explanations for {cls.key}")

        scores = [data["score"] for data in row["contributions"].values()]
        abs_sum = sum(abs(s) for s in scores)
        abs_max = max(abs(s) for s in scores)

        overall_reasoning = row["reasoning"]
        overall_confidence = row["confidence"]

        return pd.DataFrame([
            {
                "feature_name": feature,
                "score": data["score"],
                "score_normalised_sum": data["score"] / abs_sum if abs_sum != 0 else data["score"],
                "score_normalised_max": data["score"] / abs_max if abs_max != 0 else data["score"],
                "reasoning": data["reasoning"],
                "overall_reasoning": overall_reasoning,
                "overall_confidence": overall_confidence
            }
            for feature, data in row["contributions"].items()
        ])

    def explain_global_df(self, ctx: ModelContext, model=None) -> pd.DataFrame:
        raise NotImplementedError

    def explain_local_df(self, ctx: ModelContext, model=None) -> pd.DataFrame:
        raise NotImplementedError

    def explain_local_json(self, ctx: ModelContext) -> list[dict[str, Any]]:
        """Processes the CoR local explanations
        Args:
            ctx: ModelContext object

        Returns:
            list[dict[str, Any]]: list of dicts containing local explanations

        Raises:
            MissingDataError: if data.x is not provided
        """
        if ctx.data.y_exp is None or ctx.data.y_pred is None:
            raise MissingDataError("CoRAdapter requires y_exp and y_pred in ModelData")
        if ctx.response_schema is None:
            raise MissingDataError(
                "CoRAdapter requires a ResponseSchema in ModelContext")

        instance_id_col = ctx.data.data_schema.instance_id_column
        prediction_col = ctx.data.data_schema.prediction_column
        y_pred = ctx.data.y_pred
        output = []

        for _, row in ctx.data.y_exp.iterrows():
            raw = json.loads(row["explanation"])
            instance_id = row[instance_id_col]
            match = y_pred.loc[y_pred[instance_id_col] == instance_id, prediction_col]
            if match.empty:
                continue
            prediction = match.iloc[0].item() if hasattr(match.iloc[0], 'item') else match.iloc[0]
            response = self._reformat(raw, ctx.response_schema, prediction)
            output.append(response.to_dict(instance_id_col, instance_id))
        return output

    def _reformat(self, raw: dict, schema: ResponseSchema,
                  prediction: str) -> LLMResponse:
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
            prediction=prediction,
            confidence=raw["confidence"],
            contributions=contributions,
            raw=raw,
        )