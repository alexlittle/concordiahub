"""
Tests for dataset_helpers

Covers:
- render_metric_barchart
- generate_metric_card
"""



import pandas as pd
import pytest
import plotly.graph_objects as go
from unittest.mock import patch

from concordiahub.dashboard.utils.dataset_helpers import render_metric_barchart, generate_metric_card

@pytest.fixture
def metrics_df(lr_model_factory):
    """Fixture for metrics dataframe"""
    model_a = lr_model_factory(
        model_display_name="Logistic Regression",
        metrics=pd.Series({"accuracy": 0.83, "f1_weighted": 0.81}),
    )
    model_b = lr_model_factory(
        model_display_name="Random Forest",
        metrics=pd.Series({"accuracy": 0.91, "f1_weighted": 0.89}),
    )
    return pd.DataFrame(
        [model_a.store.load_series(), model_b.store.load_series()],
        index=[
            model_a.metadata["model_display_name"],
            model_b.metadata["model_display_name"],
        ],
    )

#----------------------
# Tests for render_metric_barchart
#-----------------------
def test_returns_figure_with_correct_bar_data(metrics_df):
    """Check correct data appears in the figure"""
    with patch("concordiahub.strings.METRICS_NAMES", {"accuracy": "Accuracy"}), \
         patch("concordiahub.config.PLOT_PRIMARY_COLOR", "#1f77b4"), \
         patch("concordiahub.dashboard.utils.plotly_theme.THEME", {"highlight_color": "#ff0000"}), \
         patch("concordiahub.dashboard.utils.plotly_theme.apply_theme", side_effect=lambda fig, show_title: fig):
        fig = render_metric_barchart(metrics_df, "accuracy")

    assert isinstance(fig, go.Figure)
    bar_trace = fig.data[0]
    assert list(bar_trace.y) == list(metrics_df["accuracy"])
    assert list(bar_trace.x) == list(metrics_df.index)

def test_hline_marks_max_value(metrics_df):
    """Check hlines appear as max value in the plot"""
    with patch("concordiahub.strings.METRICS_NAMES", {"accuracy": "Accuracy"}), \
         patch("concordiahub.config.PLOT_PRIMARY_COLOR", "#1f77b4"), \
         patch("concordiahub.dashboard.utils.plotly_theme.THEME", {"highlight_color": "#ff0000"}), \
         patch("concordiahub.dashboard.utils.plotly_theme.apply_theme", side_effect=lambda fig, show_title: fig):
        fig = render_metric_barchart(metrics_df, "accuracy")

    max_val = metrics_df["accuracy"].max()
    hlines = [shape for shape in fig.layout.shapes if shape.type == "line"]
    assert len(hlines) == 1
    assert hlines[0].y0 == pytest.approx(max_val)
    assert hlines[0].y1 == pytest.approx(max_val)

def test_show_title_false_hides_title(metrics_df):
    """Checks title not shown in plot"""
    with patch("concordiahub.strings.METRICS_NAMES", {"accuracy": "Accuracy"}), \
         patch("concordiahub.config.PLOT_PRIMARY_COLOR", "#1f77b4"), \
         patch("concordiahub.dashboard.utils.plotly_theme.THEME", {"highlight_color": "#ff0000"}), \
         patch("concordiahub.dashboard.utils.plotly_theme.apply_theme", side_effect=lambda fig, show_title: fig):
        fig = render_metric_barchart(metrics_df, "accuracy", show_title=False)

    assert fig.layout.title.text is None


#----------------------
# Tests for generate_metric_card
#-----------------------
def test_renders_expected_html_structure():
    """Checks HTML structure of the metric card"""
    html = generate_metric_card("Accuracy", "0.83", "/models/lr")
    assert 'class="xl-metric-card"' in html
    assert 'href="/models/lr"' in html
    assert 'class="xl-metric-label">Accuracy<' in html
    assert 'class="xl-metric-value">0.83<' in html

def test_handles_numeric_value():
    html = generate_metric_card("Score", 0.95, "#")
    assert "0.95" in html