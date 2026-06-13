
from xailens.adapters.viz.viz_base import VizAdapter
from xailens.registry.viz_registry import register_viz
from xailens import strings, config


@register_viz("tabular")
class TabularAdapter(VizAdapter):
    display_name = strings.TABULAR_DISPLAY_NAME
    compatible_explanations = config.TABULAR_COMPATIBLE_XAI

    def render_local(self, explanation, axis_range=None):
        # drop unnecessary columns
        explanation = explanation.drop(columns=["score", "score_normalised_sum"])
        # rename for display
        explanation = explanation.rename(
            columns={"feature_name": "Feature",
                     "score_normalised_max": "Score (1 to -1)",
                     "reasoning": "Reasoning"})
        return explanation.sort_values("Score (1 to -1)", ascending=False)

    def render_global(self, explanations):
        # Placeholder function
        return