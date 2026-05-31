import plotly.graph_objects as go

from xai_compare.adapters.viz.viz_base import VizAdapter
from xai_compare.registry.viz_registry import register_viz
from xai_compare import strings, config
from xai_compare.services.preprocess import normalise_values


@register_viz("forceplot")
class ForcePlotVizAdapter(VizAdapter):

    display_name = strings.FORCE_PLOT_DISPLAY_NAME
    compatible_explanations = config.FORCE_PLOT_COMPATIBLE_XAI

    def render_local(self, explanation, sample_id_col, axis_range=None):
        features = normalise_values(explanation, sample_id_col)

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
        if axis_range:
            fig.update_layout(xaxis=dict(range=axis_range))
            
        return fig

    def render_global(self, explanations):
        # Placeholder function
        return