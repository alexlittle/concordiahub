import plotly.graph_objects as go
from xai_compare.adapters.viz.viz_base import VizAdapter
from xai_compare.registry.viz_registry import register_viz

@register_viz("waterfall")
class WaterfallVizAdapter(VizAdapter):

    display_name = "Waterfall"
    compatible_explanations = ["shap_tree","coefficients"]

    def render_local(self, explanation)  -> go.Figure:
        features = explanation.drop(["sample_uuid", "class"], errors="ignore")

        # Sort by absolute value descending
        features = features.reindex(features.abs().sort_values(ascending=True).index)

        # Normalise by sum of absolute values
        features = features / features.abs().sum()

        fig = go.Figure(go.Waterfall(
            orientation="h",
            measure=["relative"] * len(features),
            y=features.index.tolist(),
            x=features.values.tolist(),
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig.update_layout(
            title="Feature Contributions (normalised)",
            xaxis_title="Normalised contribution",
        )
        return fig

    def render_global(self, explanations):
        # Placeholder function
        return