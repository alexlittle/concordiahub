
from xailens.registry.comparison_registry import register_comparison
from xailens.adapters.comparison.comparison_base import ComparisonAdapter
from xailens import config
from xailens.entities import context


@register_comparison("tree_shap", "cor")
@register_comparison("cor", "cor")
@register_comparison("coefficients", "cor")
class AlignmentScaleComparisonAdapter(ComparisonAdapter):

    default_visualisation = config.ALIGNMENT_SCALE_COMPARISON_VISUALISATION

    visualisations_available = config.ALIGNMENT_SCALE_COMPARISON_VISUALISATION_OPTIONS

    def render_comparison(self, ctx: context.ComparisonContext, viz_key:str=default_visualisation):
        option = self.visualisations_available[viz_key]
        return getattr(self, option.method)(ctx)