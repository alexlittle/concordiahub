import pandas as pd
import plotly.express as px


from xailens.dashboard.utils import plotly_theme
from xailens import strings


def render_accuracy_barchart(metrics: pd.DataFrame):
    fig = px.bar(
        metrics,
        x=metrics.index,
        y="accuracy",
        title=strings.PERF_GRAPH_ACC_TITLE,
        labels={"x": "Model", "accuracy": "Accuracy"},
        color=metrics.index,
        color_discrete_sequence=plotly_theme.THEME["color_sequence"],
    )
    fig.update_xaxes(title_text="Model")
    return plotly_theme.apply_theme(fig)


def render_f1_weighted_barchart(metrics: pd.DataFrame):
    fig = px.bar(
        metrics,
        x=metrics.index,
        y="f1_weighted",
        title=strings.PERF_GRAPH_F1_WEIGHTED_TITLE,
        labels={"x": "Model", "f1_weighted": "F1 weighted"},
        color=metrics.index,
        color_discrete_sequence=plotly_theme.THEME["color_sequence"],
    )
    fig.update_xaxes(title_text="Model")
    return plotly_theme.apply_theme(fig)


def render_precision_weighted_barchart(metrics: pd.DataFrame):
    fig = px.bar(
        metrics,
        x=metrics.index,
        y="precision_weighted",
        title=strings.PERF_GRAPH_PRECISION_WEIGHTED_TITLE,
        labels={"x": "Model", "precision_weighted": "Precision weighted"},
        color=metrics.index,
        color_discrete_sequence=plotly_theme.THEME["color_sequence"],
    )
    fig.update_xaxes(title_text="Model")
    return plotly_theme.apply_theme(fig)


def render_recall_weighted_barchart(metrics: pd.DataFrame):
    fig = px.bar(
        metrics,
        x=metrics.index,
        y="recall_weighted",
        title=strings.PERF_GRAPH_RECALL_WEIGHTED_TITLE,
        labels={"x": "Model", "recall_weighted": "Recall weighted"},
        color=metrics.index,
        color_discrete_sequence=plotly_theme.THEME["color_sequence"],
    )
    fig.update_xaxes(title_text="Model")
    return plotly_theme.apply_theme(fig)



