import plotly.graph_objects as go
import pandas as pd

from xailens.adapters.viz.viz_base import VizAdapter
from xailens.registry.viz_registry import register_viz
from xailens import strings, config


@register_viz("forceplot")
class ForcePlotVizAdapter(VizAdapter):

    display_name = strings.FORCE_PLOT_DISPLAY_NAME
    compatible_explanations = config.FORCE_PLOT_COMPATIBLE_XAI

    def render_local(self, explanation: pd.DataFrame, axis_range=None):
        explanation = explanation.sort_values("score_normalised_max", ascending=True)
        colors = [config.PLOT_LEFT_COLOR if v < 0 else config.PLOT_RIGHT_COLOR
                  for v in explanation["score_normalised_max"]]
        fig = go.Figure(go.Bar(
            orientation="h",
            y=explanation["feature_name"].tolist(),
            x=explanation["score_normalised_max"].tolist(),
            marker_color=colors,
        ))
        fig.update_layout(
            title=strings.FORCE_PLOT_TITLE,
            xaxis_title=strings.FORCE_PLOT_X_AXIS,
            xaxis=config.PLOT_ZERO_LINE,
            showlegend=False,
        )
        if axis_range:
            fig.update_layout(xaxis={"range": axis_range})

        return fig

    def render_global(self, explanations):
        # Placeholder function
        return