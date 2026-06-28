import plotly.graph_objects as go
import pandas as pd

from xailens.dashboard.utils import plotly_theme


def render_features_heatmap(df_exp: pd.DataFrame):
    model_cols = [col for col in df_exp.columns if col != "feature_name"]
    llm_cols = [col for col in model_cols if df_exp[col].isna().all()]
    data_cols = [col for col in model_cols if col not in llm_cols]

    x_labels = data_cols + [f"{col} *" for col in llm_cols]


    fig = go.Figure()

    # main heatmap for models with data
    if data_cols:
        z_main = [
            [row[col] for col in data_cols] + [None] * len(llm_cols)
            for _, row in df_exp.iterrows()
        ]
        fig.add_trace(go.Heatmap(
            z=z_main,
            x=x_labels,
            y=df_exp["feature_name"].tolist(),
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar={"title": "Importance"},
            hovertemplate="<b>%{x}</b><br>%{y}<br>Importance: %{z:.4f}<extra></extra>",
            hoverongaps=False,
        ))

    # grey overlay for LLM columns
    if llm_cols:
        z_grey = [
            [None] * len(data_cols) + [0.5] * len(llm_cols)
            for _ in range(len(df_exp))
        ]
        fig.add_trace(go.Heatmap(
            z=z_grey,
            x=x_labels,
            y=df_exp["feature_name"].tolist(),
            colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
            showscale=False,
            hovertemplate="<b>%{x}</b><br>%{y}<br>Explanation not available<extra></extra>",
            hoverongaps=False,
        ))

    fig = plotly_theme.apply_theme(fig)
    fig.update_layout(
        title="",
        xaxis={"side": "top"},
        yaxis={"autorange":"reversed", "range": None},
        height=max(400, len(df_exp) * 30),
        width=max(400, len(model_cols) * 200),
        hovermode="closest",
    )
    return fig