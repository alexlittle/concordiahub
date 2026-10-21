"""
Helpers for the dataset page
"""

import pandas as pd
import plotly
import plotly.express as px


from concordiahub.dashboard.utils import plotly_theme
from concordiahub import strings, config
from concordiahub.dashboard.assets.icons import arrow_icon


def render_metric_barchart(
        metrics: pd.DataFrame,
        metric_name: str,
        show_title: bool=True) -> plotly.graph_objects.Figure:
    """
    Renders a barchart of the metrics

    Args:
        metrics (pd.DataFrame): dataframe of metrics
        metric_name (str): name of the metric to display
        show_title (bool, optional): whether to display the title of the metric. Defaults to True

    Returns:
        plotly.graph_objects.Figure: plotly graph object
    """

    # find the best model
    max_val = metrics[metric_name].max()

    fig = px.bar(
        metrics,
        x=metrics.index,
        y=metric_name,
        title=strings.METRICS_NAMES[metric_name] if show_title else None,
        labels={"x": "Model", metric_name: strings.METRICS_NAMES[metric_name]},
    )
    fig.update_traces(marker_color=config.PLOT_PRIMARY_COLOR)

    fig.add_hline(
        y=max_val,
        line_dash="dash",
        line_color=plotly_theme.THEME["highlight_color"],
        annotation_text=f"Highest Score: {max_val:.2f}",
        annotation_position="top left",
        annotation_font={
            "size": 14,
            "color": plotly_theme.THEME["highlight_color"],
        },
    )
    fig.update_xaxes(title_text="Model")

    return plotly_theme.apply_theme(fig, show_title=show_title)


def generate_metric_card(title:str, value:str, link:str)-> str:
    """
    Generates HTML code for a metrics card

    Args:
        title (str): title of the metric
        value (str): value of the metric
        link (str): where the metric card will link to

    Returns:
        str: HTML code for the metric card
    """
    return f"""
    <div class="xl-metric-card" title="View {title.lower()}">
        <a href="{link}" target="_self">
            <div class="xl-metric-label">{title}</div>
            <div class="xl-metric-value-container">
                <span class="xl-metric-value">{value}</span>
                {arrow_icon("test")}
            </div>
        </a>
    </div>
"""

