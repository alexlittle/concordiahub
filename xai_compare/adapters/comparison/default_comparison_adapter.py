from xai_compare.registry.comparison_registry import register_comparison
from xai_compare.adapters.comparison.comparison_base import ComparisonAdapter
from xai_compare import context


@register_comparison("coefficients", "shap_tree")
@register_comparison("coefficients", "coefficients")
@register_comparison("shap_tree", "shap_tree")
class DefaultComparisonAdapter(ComparisonAdapter):
    """A default comparison adapter for shap_tree and coefficients"""

    default_visualisation = "dumbbell"

    visualisations_available = {
        "diff_bar": context.VizOption(display_name="Difference (bar)",
                              method="viz_diff_bar"),
        "dumbbell": context.VizOption(display_name="Dumbbell", method="viz_dumbbell"),
        "alignment_scale": context.VizOption(display_name="Alignment Scale",
                                     method="viz_alignment_scale"),
    }

    def render_comparison(self, ctx: context.ComparisonContext, viz_key:str=default_visualisation):
        option = self.visualisations_available[viz_key]
        return getattr(self, option.method)(ctx)
