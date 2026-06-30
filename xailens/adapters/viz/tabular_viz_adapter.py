import pandas as pd
from typing import Any

from xailens.adapters.viz.viz_base import VizAdapter
from xailens.registry.viz_registry import register_viz
from xailens import strings, config


@register_viz("tabular")
class TabularAdapter(VizAdapter):
    display_name = strings.TABULAR_DISPLAY_NAME
    compatible_explanations = config.TABULAR_COMPATIBLE_XAI

    def render(self, explanation: pd.DataFrame, axis_range=None) -> list[Any]:

        explanation = self._top_features(explanation)
        explanation = explanation.sort_values("score_normalised_max", ascending=True)
        parts = []

        # get the high level explanation if available
        if "overall_reasoning" in explanation.columns:
            overall_reasoning = explanation["overall_reasoning"].iloc[0]
            overall_confidence = explanation["overall_confidence"].iloc[0]

            overall_text = f"""
                    <em>Overall Reasoning</em>: {overall_reasoning} <br/>
                    <em>Overall Confidence</em>: {overall_confidence} <br/>"""
            parts.append(overall_text)

            final_columns = {"feature_name": "Feature",
                     "score_normalised_max": strings.TABULAR_SCORE_TITLE,
                     "reasoning": "Reasoning"}
            # drop unnecessary columns
            explanation = explanation.drop(columns=["score",
                                                    "score_normalised_sum",
                                                    "overall_reasoning",
                                                    "overall_confidence"])
        else:
            final_columns = {"feature_name": "Feature",
                             "score_normalised_max": strings.TABULAR_SCORE_TITLE}
            # drop unnecessary columns
            explanation = explanation.drop(columns=["score",
                                                    "score_normalised_sum",
                                                    "reasoning"])



        # rename for display
        explanation = explanation.rename(columns=final_columns)
        parts.append(explanation.sort_values(strings.TABULAR_SCORE_TITLE, ascending=False))


        return parts
