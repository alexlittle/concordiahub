import pandas as pd
import plotly
import plotly.express as px


from xailens.dashboard.utils import plotly_theme
from xailens import strings


def render_metric_barchart(metrics: pd.DataFrame, metric_name: str) -> plotly.graph_objects.Figure:
    fig = px.bar(
        metrics,
        x=metrics.index,
        y=metric_name,
        title=strings.PERF_GRAPH_ACC_TITLE,
        labels={"x": "Model", metric_name: strings.METRICS_NAMES[metric_name]},
        color=metrics.index,
        color_discrete_sequence=plotly_theme.THEME["color_sequence"],
    )
    fig.update_xaxes(title_text="Model")
    return plotly_theme.apply_theme(fig)




