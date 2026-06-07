import plotly.graph_objects as go
import pandas as pd

from xai_compare.adapters.viz.viz_base import VizAdapter
from xai_compare.registry.viz_registry import register_viz
from xai_compare import strings, config


@register_viz("forceplot")
class ForcePlotVizAdapter(VizAdapter):

    display_name = strings.FORCE_PLOT_DISPLAY_NAME
    compatible_explanations = config.FORCE_PLOT_COMPATIBLE_XAI

    def render_local(self, explanation, sample_id_col, axis_range=None):
        features = explanation.drop(index=sample_id_col, errors='ignore')
        features = pd.to_numeric(features, errors='coerce')

        colors = [config.PLOT_LEFT_COLOR if v < 0 else config.PLOT_RIGHT_COLOR for v in features.values]

        fig = go.Figure(go.Bar(
            orientation="h",
            y=features.index.tolist(),
            x=features.values.tolist(),
            marker_color=colors,
        ))

        fig.update_layout(
            title=strings.FORCE_PLOT_TITLE,
            xaxis_title=strings.FORCE_PLOT_X_AXIS,
            xaxis=config.PLOT_ZERO_LINE,
            showlegend=False,
        )
        if axis_range:
            fig.update_layout(xaxis={"range":axis_range})
            
        return fig

    def render_global(self, explanations):
        # Placeholder function
        return