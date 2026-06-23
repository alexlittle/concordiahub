import pandas as pd
import plotly.express as px

from xailens.dashboard.utils.render import plotly_theme


def render_accuracy_barchart(metrics: pd.DataFrame):
    fig = px.bar(
        metrics,
        x=metrics.index,
        y="accuracy",
        title="Accuracy",
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
        title="F1 weighted",
        labels={"x": "Model", "f1_weighted": "F1 weighted"},
        color=metrics.index,
        color_discrete_sequence=plotly_theme.THEME["color_sequence"],
    )
    fig.update_xaxes(title_text="Model")
    return plotly_theme.apply_theme(fig)