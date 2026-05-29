from xai_compare.adapters.viz.viz_base import VizAdapter
from xai_compare.adapters.explainers.treeshap_adapter import TreeSHAPAdapter
from xai_compare.registry.viz_registry import register_viz


@register_viz("beeswarm")
class BeeswarmVizAdapter(VizAdapter):

    display_name = "Beeswarm"
    compatible_explanations = ["shap_tree"]

    def render(self, explanation):
        pass