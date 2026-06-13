from typing import Any

import plotly.graph_objects as go
from xailens.adapters.viz.viz_base import VizAdapter
from xailens.registry.viz_registry import register_viz
from xailens import strings, config

@register_viz("waterfall")
class WaterfallVizAdapter(VizAdapter):

    display_name = strings.WATERFALL_DISPLAY_NAME
    compatible_explanations = config.WATERFALL_COMPATIBLE_XAI

    def render_local(self, explanation, axis_range=None) -> list[Any]:
        explanation.sort_values("score_normalised_max", inplace=True, ascending=True)
        fig = go.Figure(go.Waterfall(
            orientation="h",
            measure=["relative"] * len(explanation["feature_name"]),
            y=explanation["feature_name"].tolist(),
            x=explanation["score"].tolist(),
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig.update_layout(
            title=strings.WATERFALL_TITLE,
            xaxis_title=strings.WATERFALL_X_AXIS,
        )
        return [fig]

    def render_global(self, explanations):
        # Placeholder function
        return