import plotly.graph_objects as go

from xai_compare.context import ComparisonContext
from xai_compare.registry.comparison_registry import register_comparison
from xai_compare.adapters.comparison.comparison_base import ComparisonAdapter
from xai_compare.dashboard.utils.formatting import format_model_name
from xai_compare import config, context


@register_comparison("coefficients", "shap_tree")
class CoefficientTreeShapComparison(ComparisonAdapter):
    """For comparing coefficients with shap tree"""


    def _viz_diff_bar(self, features_a, features_b, run_a, run_b, axis_range=None):
        raise NotImplementedError

    def _viz_dumbbell(self, features_a, features_b, run_a, run_b, axis_range=None):
        raise NotImplementedError

    def _viz_alignment_scale(self, ctx: ComparisonContext ):
        diff = ctx.features_a.sub(ctx.features_b, fill_value=0)
        diff = diff.sort_values()

        colors = [config.PLOT_LEFT_COLOR if v < 0 else config.PLOT_RIGHT_COLOR for v in diff]

        fig = go.Figure(go.Bar(
            x=diff.values,
            y=diff.index,
            orientation='h',
            marker_color=colors,
        ))

        label_a = format_model_name(ctx.run_a)
        label_b = format_model_name(ctx.run_b)

        fig.update_layout(
            title=f"Explanation Difference: {label_a} − {label_b}",
            xaxis_title=f"Attribution difference ({label_a} − {label_b})",
            xaxis=config.PLOT_ZERO_LINE,
            yaxis_title="Feature",
            height=400 + len(diff) * 15,
        )
        if ctx.axis_range:
            fig.update_layout(xaxis={"range": ctx.axis_range})
        return fig

    visualisations_available = {
        "Difference (bar)": _viz_diff_bar,
        "Dumbbell": _viz_dumbbell,
        "Alignment Scale": _viz_alignment_scale
    }

    def render_comparison(self, ctx: context.ComparisonContext, viz_type):

        return self.visualisations_available[viz_type](self, ctx)
