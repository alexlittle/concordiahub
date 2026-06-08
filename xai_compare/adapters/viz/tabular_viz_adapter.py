
from xai_compare.adapters.viz.viz_base import VizAdapter
from xai_compare.registry.viz_registry import register_viz
from xai_compare import strings, config


@register_viz("tabular")
class TabularAdapter(VizAdapter):
    display_name = strings.TABULAR_DISPLAY_NAME
    compatible_explanations = config.TABULAR_COMPATIBLE_XAI

    def render_local(self, explanation, sample_id_col, axis_range=None):
        print(explanation)
        return explanation

    def render_global(self, explanations):
        # Placeholder function
        return