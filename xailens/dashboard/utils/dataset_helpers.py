import pandas as pd
import plotly
import plotly.express as px


from xailens.dashboard.utils import plotly_theme
from xailens import strings, config


def render_metric_barchart(metrics: pd.DataFrame, metric_name: str, show_title: bool=True) -> plotly.graph_objects.Figure:

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




