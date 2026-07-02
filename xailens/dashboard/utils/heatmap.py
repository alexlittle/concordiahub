import plotly.graph_objects as go
import pandas as pd

from xailens.dashboard.utils import plotly_theme

def render_features_heatmap(df_exp: pd.DataFrame):
    model_cols = sorted(col for col in df_exp.columns if col != "feature_name")
    llm_cols = {col for col in model_cols if df_exp[col].isna().all()}
    x_labels = [f"{col} *" if col in llm_cols else col for col in model_cols]
    y_labels = df_exp["feature_name"].tolist()

    z_main = [
        [row[col] if pd.notna(row[col]) else None for col in model_cols]
        for _, row in df_exp.iterrows()
    ]
    missing_mask = [
        [pd.isna(row[col]) for col in model_cols]
        for _, row in df_exp.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z_main,
        x=x_labels,
        y=y_labels,
        colorscale="Blues",
        zmin=0,
        zmax=1,
        colorbar={"title": "Importance"},
        hovertemplate="<b>%{x}</b><br>%{y}<br>Importance: %{z:.4f}<extra></extra>",
        hoverongaps=False,
    ))

    if any(any(row_mask) for row_mask in missing_mask):
        z_grey = [
            [1 if is_missing else None for is_missing in row_mask]
            for row_mask in missing_mask
        ]
        fig.add_trace(go.Heatmap(
            z=z_grey,
            x=x_labels,
            y=y_labels,
            colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
            showscale=False,
            hovertemplate="<b>%{x}</b><br>%{y}<br>No data<extra></extra>",
            hoverongaps=False,
        ))

    fig = plotly_theme.apply_theme(fig)
    fig.update_layout(
        title="",
        xaxis={"side": "top"},
        yaxis={"autorange": "reversed", "range": None},
        height=max(400, len(df_exp) * 30),
        width=max(400, len(model_cols) * 200),
        hovermode="closest",
    )
    return fig