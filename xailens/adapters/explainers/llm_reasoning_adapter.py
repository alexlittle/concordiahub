"""
LLMReasoningAdapter class - for processing explanations from LLMs
"""

import json

import pandas as pd
from typing import Any, Optional
from dataclasses import dataclass

from xailens.entities.context import ModelContext, ResponseSchema
from xailens.exceptions import MissingDataError
from xailens.registry.xai_registry import register_xai
from xailens.adapters.explainers.xaimethod_base import XAIMethodAdapter
from xailens import strings, config
from xailens.entities.model import Model


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

@register_xai("llm_reasoning")
class LLMReasoningAdapter(XAIMethodAdapter):


    display_name = strings.LLM_REASONING_DISPLAY_NAME
    display_notes = strings.LLM_REASONING_NOTES
    default_visualisation = config.LLM_REASONING_DEFAULT_VIZ
    compatible_visualisations = config.LLM_REASONING_COMPATIBLE_VIZ

    # uses a json file for explanation storage
    explanation_file_type: str = "json"

    @classmethod
    def load_global_explanation(cls, model: Model) -> pd.DataFrame:
        """
        Returns the global explanation for the specified model

        Args:
            model (Model): a Model object to get the explanation for

        Returns:
            pd.DataFrame: dataframe containing the global explanation, with the following columns:
                feature_name, score, score_normalised_sum, score_normalised_max
        """
        df = model.store.load_dataframe(f"exp_global_{cls.key}")
        abs_importance = df["importance"].abs()
        abs_sum = abs_importance.sum()
        abs_max = abs_importance.max()
        return pd.DataFrame({
            "feature_name": df["feature"],
            "score": df["importance"],
            "score_normalised_sum": abs_importance / abs_sum if abs_sum != 0 else abs_importance,
            "score_normalised_max": abs_importance / abs_max if abs_max != 0 else abs_importance,
        })

    @classmethod
    def load_local_explanation(cls, model: Model, instance_id: str) -> pd.DataFrame:
        """
        Returns the local explanation for a given instance

        Args:
            model (Model): a Model object
            instance_id (str): identifier for the instance to get the local explanation for

        Returns:
            pd.DataFrame: dataframe containing the local explanation, with the following columns:
                feature_name, score, score_normalised_sum, score_normalised_max, reasoning, overall_reasoning, overall_confidence

        Raises:
            ValueError: if instance_id is not provided
        """
        records = model.store.load_json(f"exp_local_{cls.key}")
        instance_id_col = model.get(config.DATA_SCHEMA_ID_COLUMN)

        row = next((r for r in records if r[instance_id_col] == instance_id), None)
        if row is None:
            raise ValueError(
                f"Instance '{instance_id}' not found in explanations for {cls.key}")

        # make sure data[score] treaded as a proper float value
        contributions = list(row["contributions"].items())
        scores = [float(data["score"]) for _, data in contributions]

        abs_sum = sum(abs(s) for s in scores)
        abs_max = max(abs(s) for s in scores)

        overall_reasoning = row["reasoning"]
        overall_confidence = row["confidence"]

        return pd.DataFrame([
            {
                "feature_name": feature,
                "score": score,
                "score_normalised_sum": score / abs_sum if abs_sum != 0 else score,
                "score_normalised_max": score / abs_max if abs_max != 0 else score,
                "reasoning": data["reasoning"],
                "overall_reasoning": overall_reasoning,
                "overall_confidence": overall_confidence
            }
            for (feature, data), score in zip(contributions, scores)
        ])

    def explain_global_json(self, ctx: ModelContext, _model: Optional[Any] =None) -> pd.DataFrame:
        """Generates the global explanations
        Uses the total number of instances of a mentioned feature divided by the
        total number of instances

        Args:
            ctx (ModelContext) : ModelContext object
            _model (Optional[Any]) : Unused

        Returns:
            pd.DataFrame: dataframe containing global explanations, with the following columns:
                feature, importance, occurrence_count, occurrence_rate, n_instances

        Raises:
            MissingDataError: if data.y_exp is not provided
        """
        if ctx.data.y_exp is None:
            raise MissingDataError("LLMReasoningAdapter requires y_exp in ModelData")
        if ctx.response_schema is None:
            raise MissingDataError("LLMReasoningAdapter requires a ResponseSchema in ModelContext")

        results = {}
        n_instances = 0

        for _, row in ctx.data.y_exp.iterrows():
            raw = json.loads(row["explanation"])
            n_instances += 1

            response = self._reformat(raw, ctx.response_schema, prediction=None)

            for feature, contrib in response.contributions.items():
                if feature not in results:
                    results[feature] = {"sum_abs_score": 0, "occurrence_count": 0}
                results[feature]["sum_abs_score"] += abs(float(contrib.score))
                results[feature]["occurrence_count"] += 1

        rows = []
        for feature, vals in results.items():
            rows.append({
                "feature": feature,
                "importance": vals["sum_abs_score"] / n_instances,
                "occurrence_count": vals["occurrence_count"],
                "occurrence_rate": vals["occurrence_count"] / n_instances,
                "n_instances": n_instances,
            })

        return pd.DataFrame(rows)


    def explain_global_df(self, ctx: ModelContext, model=None) -> pd.DataFrame:
        """
        Function is not implemented for this class instance

        Args:
            ctx ():
            model ():

        Returns:

        Raises:
            NotImplementedError

        """
        raise NotImplementedError

    def explain_local_df(self, ctx: ModelContext, model=None) -> pd.DataFrame:
        """
        Function is not implemented for this class instance

        Args:
            ctx ():
            model ():

        Returns:

        Raises:
            NotImplementedError

        """
        raise NotImplementedError

    def explain_local_json(self, ctx: ModelContext) -> list[dict[str, Any]]:
        """
        Processes the local explanations

        Args:
            ctx (ModelContext): ModelContext object

        Returns:
            list[dict[str, Any]]: list of dicts containing local explanations

        Raises:
            MissingDataError: if data.x is not provided
        """
        if ctx.data.y_exp is None or ctx.data.y_pred is None:
            raise MissingDataError("LLMReasoningAdapter requires y_exp and y_pred in ModelData")
        if ctx.response_schema is None:
            raise MissingDataError(
                "LLMReasoningAdapter requires a ResponseSchema in ModelContext")

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

    @staticmethod
    def _reformat(raw: dict, schema: ResponseSchema, prediction: str | None) -> LLMResponse:
        """
        Private helper function to reformat an explanation in raw dict format (from JSON) into an LLMResponse object.
        This uses the DataSchema definition to process the specific dict structure into a common LLMResponse object structure.

        Args:
            raw (dict): dict
            schema (ResponseSchema): response schema:
            prediction (str|None): prediction, if defined

        Returns:
            LLMResponse
        """
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
                    reasoning=values.get("reasoning", ""),
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