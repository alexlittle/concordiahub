from typing import Any

from xailens.adapters.viz.viz_base import VizAdapter
from xailens.registry.viz_registry import register_viz
from xailens import strings, config


@register_viz("tabular")
class TabularAdapter(VizAdapter):
    display_name = strings.TABULAR_DISPLAY_NAME
    compatible_explanations = config.TABULAR_COMPATIBLE_XAI

    def render_local(self, explanation, axis_range=None) -> list[Any]:

        # get the high level explanation
        overall_reasoning = explanation["overall_reasoning"].iloc[0]
        overall_confidence = explanation["overall_confidence"].iloc[0]

        overall_text = f"""
                <em>Overall Reasoning</em>: {overall_reasoning} <br/>
                <em>Overall Confidence</em>: {overall_confidence} <br/>"""

        # drop unnecessary columns
        explanation = explanation.drop(columns=["score",
                                                "score_normalised_sum",
                                                "overall_reasoning",
                                                "overall_confidence"])

        # rename for display
        explanation = explanation.rename(
            columns={"feature_name": "Feature",
                     "score_normalised_max": "Score (1 to -1)",
                     "reasoning": "Reasoning"})



        return [overall_text, explanation.sort_values("Score (1 to -1)", ascending=False)]
