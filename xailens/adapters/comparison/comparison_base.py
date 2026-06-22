
import plotly.graph_objects as go

from abc import ABC, abstractmethod
from typing import Any


from xailens.formatting import format_model_name
from xailens.entities.context import ComparisonContext
from xailens import config
from xailens.schema import ALIGNMENT_LEVELS
from xailens.services.alignment import compute_feature_differences, score_to_level

class ComparisonAdapter(ABC):
    """Handles comparison between two XAI explanations."""

    compatible_pairs: set[frozenset[str]]
    default_visualisation = None
    visualisations_available = []

    @abstractmethod
    def render_comparison(self, ctx: ComparisonContext, viz_key:str = None) -> Any:
        raise NotImplementedError

    def get_visualisations_available(self):
        return self.visualisations_available

    def get_default_visualisation(self):
        return self.default_visualisation

    def viz_diff_bar(self, ctx: ComparisonContext):
        a = ctx.explanation_a.set_index("feature_name")["score_normalised_max"]
        b = ctx.explanation_b.set_index("feature_name")["score_normalised_max"]
        diff = a.sub(b, fill_value=0).sort_values()
        colors = [config.PLOT_LEFT_COLOR if v < 0 else config.PLOT_RIGHT_COLOR
                  for v in diff]
        fig = go.Figure(go.Bar(
            x=diff.values,
            y=diff.index,
            orientation='h',
            marker_color=colors,
        ))
        label_a = format_model_name(ctx.modeleval_a)
        label_b = format_model_name(ctx.modeleval_b)
        fig.update_layout(
            title=f"Explanation Difference: {label_a} → {label_b}",
            xaxis_title=f"Attribution difference ({label_a} → {label_b})",
            xaxis=config.PLOT_ZERO_LINE,
            yaxis_title="Feature",
            height=400 + len(diff) * 15,
        )
        diff_max = max(abs(diff.values))
        margin = diff_max * 0.15
        fig.update_layout(
            xaxis={"range": [-(diff_max + margin), diff_max + margin]})
        return fig

    def viz_dumbbell(self, ctx: ComparisonContext):
        a = ctx.explanation_a.set_index("feature_name")["score_normalised_max"]
        b = ctx.explanation_b.set_index("feature_name")["score_normalised_max"]
        features = a.index.union(b.index)
        series_a = a.reindex(features, fill_value=0)
        series_b = b.reindex(features, fill_value=0)
        midpoints = (series_a + series_b) / 2
        order = midpoints.sort_values().index
        a_vals = series_a.reindex(order)
        b_vals = series_b.reindex(order)
        label_a = format_model_name(ctx.modeleval_a)
        label_b = format_model_name(ctx.modeleval_b)
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
        """Single dot scale from mean difference - for instance list."""
        '''
        differences = self._compute_differences(ctx)
        mean_diff = sum(d for _, d in differences) / len(
            differences) if differences else 0.0
        return self._render_dot_scale(self._score_to_level(mean_diff))
        '''
        raise NotImplementedError

    def viz_alignment_scale(self, ctx: ComparisonContext) -> str:
        """Per-feature dot scale - for detail page."""
        label_a = format_model_name(ctx.modeleval_a)
        label_b = format_model_name(ctx.modeleval_b)
        matched, only_in_a, only_in_b = compute_feature_differences(
            ctx.explanation_a, ctx.explanation_b)
        matched_rows = "".join(
            f'<tr>'
            f'<td style="padding:4px 12px 4px 0;color:#0d4f4f;font-size:13px;">{feature}</td>'
            f'<td>{self._render_dot_scale(score_to_level(diff))}</td>'
            f'</tr>'
            for feature, diff in matched
        )
        absent_html = (
                self.absent_section(f"Only in {label_a}", only_in_a) +
                self.absent_section(f"Only in {label_b}", only_in_b)
        )
        return f'<table style="border-collapse:collapse;">{matched_rows}{absent_html}</table>'

    #-----------------------------
    # Private helper functions
    #-----------------------------

    @staticmethod
    def _render_dot_scale(level: int) -> str:
        active_entry = next(e for e in ALIGNMENT_LEVELS if e.level == level)
        dots = []
        for entry in ALIGNMENT_LEVELS:
            active = entry.level == level
            colour = active_entry.colour if active else "#D4E4E4"
            border = active_entry.colour if active else "#aaaaaa"
            size = "18px" if active else "12px"
            dots.append(
                f'<span title="{entry.label}" style="'
                f'display:inline-block;'
                f'width:{size};height:{size};'
                f'border-radius:50%;'
                f'background:{colour};'
                f'border: 2px solid {border};'
                f'margin: 0 4px;'
                f'vertical-align:middle;'
                f'"></span>'
            )
        return (
            f'<div style="display:flex;align-items:center;gap:4px;">'
            f'{"".join(dots)}'
            f'<span style="margin-left:8px;font-size:14px;color:{active_entry.colour};">'
            f'{active_entry.label}</span>'
            f'</div>'
        )

    def absent_section(self, title: str, items: list[tuple[str, str]]) -> str:
        if not items:
            return ""
        header = (
            f'<tr><td colspan="2" style="'
            f'padding:8px 0 4px 0;'
            f'color:#888;'
            f'font-size:12px;'
            f'border-top:1px solid #D4E4E4;'
            f'">{title}</td></tr>'
        )
        rows = "".join(
            f'<tr>'
            f'<td style="padding:4px 12px 4px 0;color:#0d4f4f;font-size:13px;">{feature}</td>'
            f'<td style="color:#888;font-size:12px;">score: {score}</td>'
            f'</tr>'
            for feature, score in items
        )
        return header + rows