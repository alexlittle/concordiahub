import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from xailens.dashboard.utils.render import plotly_theme
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


def render_global_features_heatmap(df_global_exp: pd.DataFrame):
    model_cols = [col for col in df_global_exp.columns if col != "feature_name"]
    llm_cols = [col for col in model_cols if df_global_exp[col].isna().all()]
    data_cols = [col for col in model_cols if col not in llm_cols]

    fig = go.Figure()

    # main heatmap for models with data
    if data_cols:
        fig.add_trace(go.Heatmap(
            z=df_global_exp[data_cols].values,
            x=data_cols,
            y=df_global_exp["feature_name"].tolist(),
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Importance"),
            hovertemplate="<b>%{x}</b><br>%{y}<br>Importance: %{z:.4f}<extra></extra>",
            hoverongaps=False,
        ))

    # grey overlay for LLM columns
    if llm_cols:
        grey_z = [[0.5] * len(llm_cols)] * len(df_global_exp)
        fig.add_trace(go.Heatmap(
            z=grey_z,
            x=[f"{col} *" for col in llm_cols],
            y=df_global_exp["feature_name"].tolist(),
            colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
            showscale=False,
            hovertemplate="<b>%{x}</b><br>%{y}<br>Global explanation not available<extra></extra>",
            hoverongaps=False,
        ))

    fig = plotly_theme.apply_theme(fig)
    fig.update_layout(
        title="",
        xaxis=dict(side="top"),
        yaxis=dict(autorange="reversed", range=None),
        height=max(400, len(df_global_exp) * 30),
        width=max(400, len(model_cols) * 200),
        hovermode="closest",
    )
    return fig
