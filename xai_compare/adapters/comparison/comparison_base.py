import pandas as pd
import plotly.graph_objects as go

from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

from xai_compare.dashboard.utils.formatting import format_model_name
from xai_compare.context import ComparisonContext
from xai_compare.services.preprocess import normalise_by_sum
from xai_compare import config


@dataclass
class VizOption:
    display_name: str
    method: str


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
    def render_comparison(self, ctx: ComparisonContext, viz_key:str =None) -> Any:
        raise NotImplementedError


    def viz_diff_bar(self, ctx: ComparisonContext):
        diff = ctx.features_a.sub(ctx.features_b, fill_value=0)
        diff = diff.sort_values()

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
            title=f"Explanation Difference: {label_a} − {label_b}",
            xaxis_title=f"Attribution difference ({label_a} − {label_b})",
            xaxis=config.PLOT_ZERO_LINE,
            yaxis_title="Feature",
            height=400 + len(diff) * 15,
        )
        if ctx.axis_range:
            fig.update_layout(xaxis={"range": ctx.axis_range})
        return fig

    def viz_dumbbell(self, ctx: ComparisonContext):
        features = ctx.features_a.index.union(ctx.features_b.index)

        a_vals = ctx.features_a.reindex(features, fill_value=0)
        b_vals = ctx.features_b.reindex(features, fill_value=0)

        # Sort by midpoint so the chart has a natural ordering
        midpoints = (a_vals + b_vals) / 2
        order = midpoints.sort_values().index
        a_vals = a_vals.reindex(order)
        b_vals = b_vals.reindex(order)

        label_a = format_model_name(ctx.run_a)
        label_b = format_model_name(ctx.run_b)

        fig = go.Figure()

        # Connector lines — one shape per feature
        for feature, a, b in zip(order, a_vals, b_vals):
            fig.add_shape(
                type="line",
                x0=a, x1=b,
                y0=feature, y1=feature,
                line=dict(color="grey", width=1.5),
            )

        # Dot for model A
        fig.add_trace(go.Scatter(
            x=a_vals,
            y=order,
            mode="markers",
            name=label_a,
            marker=dict(color=config.PLOT_LEFT_COLOR, size=10, symbol="circle"),
        ))

        # Dot for model B
        fig.add_trace(go.Scatter(
            x=b_vals,
            y=order,
            mode="markers",
            name=label_b,
            marker=dict(color=config.PLOT_RIGHT_COLOR, size=10,
                        symbol="circle"),
        ))

        fig.update_layout(
            title=f"Feature Attribution Comparison: {label_a} vs {label_b}",
            xaxis_title="Attribution",
            xaxis=config.PLOT_ZERO_LINE,
            yaxis_title="Feature",
            height=400 + len(order) * 20,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )

        if ctx.axis_range:
            fig.update_layout(xaxis={"range": ctx.axis_range})

        return fig

    def viz_alignment_scale(self, ctx: ComparisonContext):
        raise NotImplementedError