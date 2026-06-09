
from xailens.adapters.viz.viz_base import VizAdapter
from xailens.registry.viz_registry import register_viz
from xailens import strings, config


@register_viz("tabular")
class TabularAdapter(VizAdapter):
    display_name = strings.TABULAR_DISPLAY_NAME
    compatible_explanations = config.TABULAR_COMPATIBLE_XAI

    def render_local(self, explanation, sample_id_col, axis_range=None):
        return explanation

    def render_global(self, explanations):
        # Placeholder function
        return