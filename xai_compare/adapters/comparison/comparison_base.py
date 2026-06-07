import pandas as pd
from abc import ABC, abstractmethod
from typing import Any
import plotly.graph_objects as go

from xai_compare.dashboard.utils.formatting import format_model_name
from xai_compare.context import ComparisonContext
from xai_compare.services.preprocess import normalise_by_sum
from xai_compare import config

class ComparisonAdapter(ABC):
    """Handles comparison between two XAI explanations."""

    compatible_pairs: set[frozenset[str]]
    visualisations_available = []

    @classmethod
    def get_abs_max(cls, explanation_a: pd.Series, explanation_b: pd.Series, sample_id_column) -> float:
        """ calculate absolute max value from explanation
        required for ensuring charts use the same scales """
        features_a = normalise_by_sum(explanation_a, sample_id_column)
        features_b = normalise_by_sum(explanation_b, sample_id_column)
        a = pd.to_numeric(features_a, errors='coerce')
        b = pd.to_numeric(features_b, errors='coerce')
        diff = a.sub(b, fill_value=0)
        return diff.abs().max()

    @abstractmethod
    def render_comparison(self, ctx: ComparisonContext, viz_type) -> Any:
        raise NotImplementedError


    def viz_diff_bar(self, features_a, features_b, run_a, run_b, axis_range=None):
        raise NotImplementedError

    def viz_dumbbell(self, features_a, features_b, run_a, run_b, axis_range=None):
        raise NotImplementedError

    def viz_alignment_scale(self, ctx: ComparisonContext ):
        diff = ctx.features_a.sub(ctx.features_b, fill_value=0)
        diff = diff.sort_values()

        colors = [config.PLOT_LEFT_COLOR if v < 0 else config.PLOT_RIGHT_COLOR for v in diff]

        fig = go.Figure(go.Bar(
            x=diff.values,
            y=diff.index,
            orientation='h',
            marker_color=colors,
        ))

        label_a = format_model_name(ctx.run_a)
        label_b = format_model_name(ctx.run_b)

        fig.update_layout(
            title=f"Explanation Difference: {label_a} − {label_b}",
            xaxis_title=f"Attribution difference ({label_a} − {label_b})",
            xaxis=config.PLOT_ZERO_LINE,
            yaxis_title="Feature",
            height=400 + len(diff) * 15,
        )
        if ctx.axis_range:
            fig.update_layout(xaxis={"range": ctx.axis_range})
        return fig