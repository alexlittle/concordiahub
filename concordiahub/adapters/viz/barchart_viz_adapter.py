"""
BarChartVizAdapter implementation of VizAdapter
"""

from typing import Any, Optional

import plotly.graph_objects as go
import pandas as pd

from concordiahub.adapters.viz.viz_base import VizAdapter
from concordiahub.registry.viz_registry import register_viz
from concordiahub import strings, config


@register_viz("barchart")
class BarChartVizAdapter(VizAdapter):
    """
    BarChartVizAdapter implementation of VizAdapter
    """

    display_name = strings.BARCHART_DISPLAY_NAME
    compatible_explanations = config.BARCHART_COMPATIBLE_XAI

    def render(self,
               explanation: pd.DataFrame,
               axis_range: Optional[tuple[float, float]]=None) -> list[Any]:
        """
        Render the bar chart visualization

        Args:
            explanation (pd.DataFrame): explanation dataframe
            axis_range (Optional[tuple[float, float]]): (min, max) bounds for
                the x-axis. If None, Plotly auto-scales the axis.

        Returns:
            list[Any]: list containing the plotly figure

        """
        # filter to show only the largest contributors, not all
        explanation = self._top_features(explanation)

        explanation = explanation.sort_values("score_normalized_max",
                                              ascending=True)
        colors = [config.PLOT_SECONDARY_COLOR if v < 0 else config.PLOT_PRIMARY_COLOR
                  for v in explanation["score_normalized_max"]]

        # use features names as y-axis labels
        feature_names = explanation["feature_name"].tolist()

        # use the total n features to determine the necessary height
        n_features = len(feature_names)

        fig = go.Figure(go.Bar(
            orientation="h",
            y=feature_names,
            x=explanation["score_normalized_max"].tolist(),
            marker_color=colors,
        ))
        fig.update_layout(
            height=max(400, n_features * config.PLOT_PER_ROW_MULTIPLIER),
            title=strings.BARCHART_TITLE,
            xaxis_title=strings.BARCHART_X_AXIS,
            xaxis=config.PLOT_ZERO_LINE,
            showlegend=False,
            yaxis={
                "tickmode": "array",
                "tickvals": list(range(len(feature_names))),
                "ticktext": feature_names,
            }
        )
        if axis_range:
            fig.update_xaxes(range=axis_range)
        return [fig]