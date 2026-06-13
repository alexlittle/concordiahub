import pandas as pd
import plotly.graph_objects as go

from abc import ABC, abstractmethod
from typing import Any


from xailens.formatting import format_model_name
from xailens.context import ComparisonContext
from xailens import config
from xailens.schema import ALIGNMENT_LEVELS

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

    def viz_alignment_summary(self, ctx: ComparisonContext) -> str:
        """Single dot scale from mean difference - for sample list."""
        differences = self._compute_differences(ctx)
        mean_diff = sum(d for _, d in differences) / len(
            differences) if differences else 0.0
        return self._render_dot_scale(self._score_to_level(mean_diff))

    def viz_alignment_scale(self, ctx: ComparisonContext) -> str:
        """Per-feature dot scale - for detail page."""
        differences = self._compute_differences(ctx)
        rows_html = "".join(
            f'<tr>'
            f'<td style="padding:4px 12px 4px 0;color:#0d4f4f;font-size:13px;">{feature}</td>'
            f'<td>{self._render_dot_scale(self._score_to_level(diff))}</td>'
            f'</tr>'
            for feature, diff in differences
        )
        return f'<table style="border-collapse:collapse;">{rows_html}</table>'

    #-----------------------------
    # Private helper functions
    #-----------------------------

    @staticmethod
    def _compute_differences(ctx: ComparisonContext) -> list[tuple[str, float]]:
        """Returns list of (feature_name, difference) for all features"""
        a = ctx.explanation_a.set_index("feature_name")["score_normalised_max"]
        b = ctx.explanation_b.set_index("feature_name")["score_normalised_max"]
        features = a.index.union(b.index)
        result = []
        for feature in features:
            a_present = feature in a.index
            b_present = feature in b.index
            if a_present and b_present:
                diff = abs(a[feature] - b[feature])
            else:
                present_value = a[feature] if a_present else b[feature]
                diff = abs(present_value)
            a_val = f"{a[feature]:.4f}" if a_present else "ABSENT"
            b_val = f"{b[feature]:.4f}" if b_present else "ABSENT"
            print(
                f"[alignment] {feature}: a={a_val}, b={b_val}, diff={diff:.4f}")
            result.append((feature, diff))
        mean_diff = sum(d for _, d in result) / len(result) if result else 0.0
        print(f"[alignment] mean_diff={mean_diff:.4f}, thresholds={config.ALIGNMENT_THRESHOLDS}")
        return result

    @staticmethod
    def _score_to_level(score: float) -> int:
        for i, threshold in enumerate(config.ALIGNMENT_THRESHOLDS):
            if score < threshold:
                return len(ALIGNMENT_LEVELS) - i
        return 1

    @staticmethod
    def _render_dot_scale(level: int) -> str:
        dots = []
        for entry in ALIGNMENT_LEVELS:
            active = entry.level == level
            color = "#1A6B6B" if active else "#D4E4E4"
            size = "18px" if active else "12px"
            dots.append(
                f'<span title="{entry.label}" style="'
                f'display:inline-block;'
                f'width:{size};height:{size};'
                f'border-radius:50%;'
                f'background:{color};'
                f'border: 2px solid #1A6B6B;'
                f'margin: 0 4px;'
                f'vertical-align:middle;'
                f'"></span>'
            )
        label_text = next(e.label for e in ALIGNMENT_LEVELS if e.level == level)
        return (
            f'<div style="display:flex;align-items:center;gap:4px;">'
            f'{"".join(dots)}'
            f'<span style="margin-left:8px;font-size:14px;color:#0d4f4f;">{label_text}</span>'
            f'</div>'
        )