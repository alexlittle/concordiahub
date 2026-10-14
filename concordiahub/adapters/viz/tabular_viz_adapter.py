"""
TabularAdapter implementation of VizAdapter
"""

import pandas as pd
from typing import Any, Optional

from concordiahub.adapters.viz.viz_base import VizAdapter
from concordiahub.registry.viz_registry import register_viz
from concordiahub import strings, config


@register_viz("tabular")
class TabularAdapter(VizAdapter):
    """
    TabularAdapter implementation of VizAdapter
    """

    display_name = strings.TABULAR_DISPLAY_NAME
    compatible_explanations = config.TABULAR_COMPATIBLE_XAI

    def render(self, explanation: pd.DataFrame, axis_range: Optional[tuple[float, float]]=None) -> list[Any]:
        """
        Render the explanation in tabular format

        Args:
            explanation (pd.DataFrame): explanation dataframe
            axis_range (Optional[tuple[float, float]]): (min, max) bounds for
                the x-axis. If None, Plotly auto-scales the axis.

        Returns:
            list[Any]: list containing the overall explanation if it exists, and a dataframe for tabular display

        """

        explanation = self._top_features(explanation)
        explanation = explanation.sort_values("score_normalized_max", ascending=True)
        parts = []

        # get the high level explanation if available
        if "overall_reasoning" in explanation.columns:
            overall_reasoning = explanation["overall_reasoning"].iloc[0]
            overall_confidence = explanation["overall_confidence"].iloc[0]

            overall_text = f"""
                    <strong>{strings.TABULAR_OVERALL_REASON_TITLE}</strong>: {overall_reasoning} <br/>
                    <strong>{strings.TABULAR_OVERALL_CONF_TITLE}</strong>: {overall_confidence} <br/>"""
            parts.append(overall_text)

            final_columns = {"feature_name": strings.TABULAR_FEATURE_TITLE,
                     "score_normalized_max": strings.TABULAR_SCORE_TITLE,
                     "reasoning": "Reasoning"}
            # drop unnecessary columns
            explanation = explanation.drop(columns=["score",
                                                    "overall_reasoning",
                                                    "overall_confidence"])
        else:
            final_columns = {"feature_name": strings.TABULAR_FEATURE_TITLE,
                             "score_normalized_max": strings.TABULAR_SCORE_TITLE}
            # drop unnecessary columns
            explanation = explanation.drop(columns=["score", "reasoning"])

        # rename for display
        explanation = explanation.rename(columns=final_columns)
        parts.append(explanation.sort_values(strings.TABULAR_SCORE_TITLE, ascending=False))

        return parts
