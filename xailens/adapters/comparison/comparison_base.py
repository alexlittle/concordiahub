import pandas as pd
import plotly.graph_objects as go

from abc import ABC, abstractmethod
from typing import Any


from xailens.dashboard.utils.formatting import format_model_name
from xailens.context import ComparisonContext
from xailens import config


class ComparisonAdapter(ABC):
    """Handles comparison between two XAI explanations."""

    compatible_pairs: set[frozenset[str]]
    default_visualisation = None
    visualisations_available = []

    @classmethod
    def get_abs_max(cls, explanation_a: pd.DataFrame, explanation_b: pd.DataFrame) -> float:
        """ calculate absolute max value from explanation
        needed to ensure charts use the same scales """
        a = explanation_a.set_index("feature_name")["score_normalised_sum"]
        b = explanation_b.set_index("feature_name")["score_normalised_sum"]
        diff = a.sub(b, fill_value=0)
        return diff.abs().max()

    @abstractmethod
    def render_comparison(self, ctx: ComparisonContext, viz_key:str = None) -> Any:
        raise NotImplementedError

    def get_visualisations_available(self):
        return self.visualisations_available

    def get_default_visualisation(self):
        return self.default_visualisation

    def viz_diff_bar(self, ctx: ComparisonContext):
        a = ctx.explanation_a.set_index("feature_name")["score_normalised_sum"]
        b = ctx.explanation_b.set_index("feature_name")["score_normalised_sum"]
        diff = a.sub(b, fill_value=0).sort_values()
        colors = [config.PLOT_LEFT_COLOR if v < 0 else config.PLOT_RIGHT_COLOR
                  for v in diff]
        fig = go.Figure(go.Bar(
            x=diff.values,
            y=diff.index,
            orientation='h',
            marker_color=colors,
        ))
        label_a = format_model_name(ctx.run_a)
        label_b = format_model_name(ctx.run_b)
        fig.update_layout(
            title=f"Explanation Difference: {label_a} → {label_b}",
            xaxis_title=f"Attribution difference ({label_a} → {label_b})",
            xaxis=config.PLOT_ZERO_LINE,
            yaxis_title="Feature",
            height=400 + len(diff) * 15,
        )
        if ctx.axis_range:
            fig.update_layout(xaxis={"range": ctx.axis_range})
        return fig

    def viz_dumbbell(self, ctx: ComparisonContext):
        a = ctx.explanation_a.set_index("feature_name")["score_normalised_sum"]
        b = ctx.explanation_b.set_index("feature_name")["score_normalised_sum"]
        features = a.index.union(b.index)
        series_a = a.reindex(features, fill_value=0)
        series_b = b.reindex(features, fill_value=0)
        midpoints = (series_a + series_b) / 2
        order = midpoints.sort_values().index
        a_vals = series_a.reindex(order)
        b_vals = series_b.reindex(order)
        label_a = format_model_name(ctx.run_a)
        label_b = format_model_name(ctx.run_b)
        fig = go.Figure()
        for feature, a, b in zip(order, a_vals, b_vals):
            fig.add_shape(
                type="line",
                x0=a, x1=b,
                y0=feature, y1=feature,
                line={"color": "grey", "width": 1.5},
            )
        fig.add_trace(go.Scatter(
            x=a_vals,
            y=order,
            mode="markers",
            name=label_a,
            marker={"color": config.PLOT_LEFT_COLOR, "size": 10,
                    "symbol": "circle"},
        ))
        fig.add_trace(go.Scatter(
            x=b_vals,
            y=order,
            mode="markers",
            name=label_b,
            marker={"color": config.PLOT_RIGHT_COLOR, "size": 10,
                    "symbol": "circle"},
        ))
        fig.update_layout(
            title=f"Feature Attribution Comparison: {label_a} vs {label_b}",
            xaxis_title="Attribution",
            xaxis=config.PLOT_ZERO_LINE,
            yaxis_title="Feature",
            height=400 + len(order) * 20,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                    "xanchor": "right", "x": 1},
        )
        if ctx.axis_range:
            fig.update_layout(xaxis={"range": ctx.axis_range})
        return fig

    def viz_alignment_scale(self, ctx: ComparisonContext):
        raise NotImplementedError