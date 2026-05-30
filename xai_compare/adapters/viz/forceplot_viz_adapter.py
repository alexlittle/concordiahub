import plotly.graph_objects as go

from xai_compare.adapters.viz.viz_base import VizAdapter
from xai_compare.registry.viz_registry import register_viz


@register_viz("forceplot")
class ForcePlotVizAdapter(VizAdapter):

    display_name = "Force Plot"
    compatible_explanations = ["shap_tree", "coefficients"]

    def render_local(self, explanation):
        features = explanation.drop(["sample_uuid", "class"], errors="ignore")

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
            title="Feature Contributions (normalised)",
            xaxis_title="Normalised contribution",
            xaxis={"zeroline": True, "zerolinewidth": 2, "zerolinecolor": "black"},
            showlegend=False,
        )
        return fig

    def render_global(self, explanations):
        # Placeholder function
        return