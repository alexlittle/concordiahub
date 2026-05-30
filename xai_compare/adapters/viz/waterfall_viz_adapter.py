
from xai_compare.adapters.viz.viz_base import VizAdapter
from xai_compare.registry.viz_registry import register_viz

@register_viz("waterfall")
class WaterfallVizAdapter(VizAdapter):

    display_name = "Waterfall"
    compatible_explanations = ["shap_tree","coefficients"]

    def render(self, explanation):
        print(explanation)
        pass