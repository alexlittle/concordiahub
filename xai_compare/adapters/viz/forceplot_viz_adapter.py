import plotly.graph_objects as go

from xai_compare.adapters.viz.viz_base import VizAdapter
from xai_compare.registry.viz_registry import register_viz
from xai_compare import strings, config

@register_viz("forceplot")
class ForcePlotVizAdapter(VizAdapter):

    display_name = strings.FORCE_PLOT_DISPLAY_NAME
    compatible_explanations = config.FORCE_PLOT_COMPATIBLE_XAI

    def render_local(self, explanation, sample_id_col):
        features = explanation.drop([sample_id_col, "class"], errors="ignore")

        # Normalise by sum of absolute values
        features = features / features.abs().sum()

        # Sort all features by value
        features = features.sort_values(ascending=True)

        colors = ["red" if v < 0 else "teal" for v in features.values]

        fig = go.Figure(go.Bar(
            orientation="h",
            y=features.index.tolist(),
            x=features.values.tolist(),
            marker_color=colors,
        ))

        fig.update_layout(
            title=strings.FORCE_PLOT_TITLE,
            xaxis_title=strings.FORCE_PLOT_X_AXIS,
            xaxis={"zeroline": True, "zerolinewidth": 2, "zerolinecolor": "black"},
            showlegend=False,
        )
        return fig

    def render_global(self, explanations):
        # Placeholder function
        return