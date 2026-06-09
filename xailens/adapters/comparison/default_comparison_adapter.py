from xailens.registry.comparison_registry import register_comparison
from xailens.adapters.comparison.comparison_base import ComparisonAdapter
from xailens import context, config


@register_comparison("coefficients", "tree_shap")
@register_comparison("coefficients", "coefficients")
@register_comparison("tree_shap", "tree_shap")
class DefaultComparisonAdapter(ComparisonAdapter):
    """A default comparison adapter for shap_tree and coefficients"""

    default_visualisation = config.DEFAULT_COMPARISON_VISUALISATION

    visualisations_available = config.DEFAULT_COMPARISON_VISUALISATION_OPTIONS

    def render_comparison(self, ctx: context.ComparisonContext, viz_key:str=default_visualisation):
        option = self.visualisations_available[viz_key]
        return getattr(self, option.method)(ctx)
