from xai_compare.registry.comparison_registry import register_comparison
from xai_compare.adapters.comparison.comparison_base import ComparisonAdapter
from xai_compare import context, config


@register_comparison("coefficients", "shap_tree")
@register_comparison("coefficients", "coefficients")
@register_comparison("shap_tree", "shap_tree")
class DefaultComparisonAdapter(ComparisonAdapter):
    """A default comparison adapter for shap_tree and coefficients"""

    default_visualisation = config.DEFAULT_COMPARISON_VISUALISATION

    visualisations_available = config.DEFAULT_COMPARISON_VISUALISATION_OPTIONS

    def render_comparison(self, ctx: context.ComparisonContext, viz_key:str=default_visualisation):
        option = self.visualisations_available[viz_key]
        return getattr(self, option.method)(ctx)
