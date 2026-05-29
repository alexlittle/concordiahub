
from xai_compare.adapters.viz.viz_base import VizAdapter
from xai_compare.adapters.explainers.treeshap_adapter import TreeSHAPAdapter
from xai_compare.registry.viz_registry import register_viz

@register_viz("waterfall")
class WaterfallVizAdapter(VizAdapter):

    display_name = "Waterfall"
    compatible_explanations = ["shap_tree"]

    def render(self, explanation):
        pass