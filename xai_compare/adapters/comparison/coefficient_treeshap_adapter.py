import plotly.graph_objects as go

from xai_compare.registry.comparison_registry import register_comparison
from xai_compare.adapters.comparison.comparison_base import ComparisonAdapter
from xai_compare.dashboard.components.formatting import format_model_name
from xai_compare import constants, config
from xai_compare.services.preprocess import normalise_values


@register_comparison("coefficients", "shap_tree")
class CoefficientTreeShapComparison(ComparisonAdapter):
    """For comparing coefficients with shap tree"""

    def render_comparison(self, explanation_a, explanation_b, run_a, run_b, axis_range=None):
        features_a = normalise_values(explanation_a, run_a.get(constants.DATA_SCHEMA_ID_COLUMN))
        features_b = normalise_values(explanation_b, run_b.get(constants.DATA_SCHEMA_ID_COLUMN))

        diff = features_a.sub(features_b, fill_value=0)
        diff = diff.sort_values()

        colors = [config.PLOT_FAVOURS_B_COLOR if v < 0 else config.PLOT_FAVOURS_A_COLOR for v in diff]

        fig = go.Figure(go.Bar(
            x=diff.values,
            y=diff.index,
            orientation='h',
            marker_color=colors,
        ))

        label_a = format_model_name(run_a)
        label_b = format_model_name(run_b)

        fig.update_layout(
            title=f"Explanation Difference: {label_a} − {label_b}",
            xaxis_title=f"Attribution difference ({label_a} − {label_b})",
            xaxis={"zeroline": True, "zerolinewidth": 2, "zerolinecolor": "black"},
            yaxis_title="Feature",
            height=400 + len(diff) * 15,
        )
        if axis_range:
            fig.update_layout(xaxis=dict(range=axis_range))

        return fig