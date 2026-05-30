from xai_compare.adapters.viz.viz_base import VizAdapter
from xai_compare.registry.viz_registry import register_viz


@register_viz("forceplot")
class ForcePlotVizAdapter(VizAdapter):

    display_name = "Force Plot"
    compatible_explanations = ["shap_tree", "coefficients"]

    def render(self, explanation):
        pass