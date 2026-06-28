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

        # filter to show only the largest contributors, not all
        explanation = self._top_features(explanation)

        explanation.sort_values("score_normalised_max", inplace=True, ascending=True)

        feature_names = explanation["feature_name"].tolist()
        # use the total n features to determine the necessary height
        n_features = len(feature_names)

        fig = go.Figure(go.Waterfall(
            orientation="h",
            measure=["relative"] * len(feature_names),
            y=feature_names,
            x=explanation["score"].tolist(),
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": config.PLOT_RIGHT_COLOR}},
            decreasing={"marker": {"color": config.PLOT_LEFT_COLOR}},
        ))
        fig.update_layout(
            height=max(400, n_features * config.PLOT_PER_ROW_MULTIPLIER),
            title=strings.WATERFALL_TITLE,
            xaxis_title=strings.WATERFALL_X_AXIS,
            yaxis={
                "tickmode": "array",
                "tickvals": list(range(len(feature_names))),
                "ticktext": feature_names
            }
        )
        return [fig]