from xai_compare.registry.comparison_registry import register_comparison
from xai_compare.adapters.comparison.comparison_base import ComparisonAdapter, VizOption
from xai_compare import context


@register_comparison("coefficients", "shap_tree")
@register_comparison("coefficients", "coefficients")
@register_comparison("shap_tree", "shap_tree")
class DefaultComparisonAdapter(ComparisonAdapter):
    """A default comparison adapter for shap_tree and coefficients"""

    visualisations_available = {
        "diff_bar": VizOption(display_name="Difference (bar)",
                              method="viz_diff_bar"),
        "dumbbell": VizOption(display_name="Dumbbell", method="viz_dumbbell"),
        "alignment_scale": VizOption(display_name="Alignment Scale",
                                     method="viz_alignment_scale"),
    }

    def render_comparison(self, ctx: context.ComparisonContext, viz_key:str ="dumbbell"):
        option = self.visualisations_available[viz_key]
        return getattr(self, option.method)(ctx)
