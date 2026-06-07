from xai_compare.registry.comparison_registry import register_comparison
from xai_compare.adapters.comparison.comparison_base import ComparisonAdapter
from xai_compare import context


@register_comparison("coefficients", "shap_tree")
@register_comparison("coefficients", "coefficients")
@register_comparison("shap_tree", "shap_tree")
class DefaultComparisonAdapter(ComparisonAdapter):
    """For comparing coefficients with shap tree"""

    visualisations_available = {
        "Difference (bar)": "viz_diff_bar",
        "Dumbbell": "viz_dumbbell",
        "Alignment Scale": "viz_alignment_scale"
    }

    def render_comparison(self, ctx: context.ComparisonContext, viz_type):
        method_name = self.visualisations_available[viz_type]
        return getattr(self, method_name)(ctx)
