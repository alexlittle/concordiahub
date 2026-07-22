"""
Function for rendering a heatmap
"""

import plotly
import plotly.graph_objects as go
import pandas as pd

from xailens.dashboard.utils import plotly_theme
from xailens import strings

def render_features_heatmap(df_exp: pd.DataFrame)-> plotly.graph_objects.Figure:
    """
    Generates a heatmap visualisation

    Args:
        df_exp (pd.DataFrame): Dataframe to build the heatmap from

    Returns:
        plotly.graph_objects.Figure: the heatmap
    """
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

    # add the importance scale
    fig.add_trace(go.Heatmap(
        z=z_main,
        x=x_labels,
        y=y_labels,
        colorscale="Blues",
        zmin=0,
        zmax=1,
        colorbar={
            "title": strings.HEATMAP_COLOUR_BAR_TITLE,
            "len": 0.75,
            "y": 0.6,
            "yanchor": "middle",
            "x": 1.02,
        },
        hovertemplate=strings.HEATMAP_HOVER_TEMPLATE,
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
            hovertemplate= strings.HEATMAP_HOVER_MISSING_TEMPLATE,
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

    if any(any(row_mask) for row_mask in missing_mask):
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=1.05, x1=1.12,
            y0=0.14, y1=0.19,
            fillcolor="lightgrey",
            line={"width": 0},
        )
        fig.add_annotation(
            text="No data",
            xref="paper", yref="paper",
            x=1.13, y=0.16,
            xanchor="left", yanchor="middle",
            showarrow=False,
            font={
                "family": plotly_theme.THEME["font_family"],
                "size": plotly_theme.THEME["axis_tick_font_size"],
                "color": plotly_theme.THEME["axis_tick_color"],
            },
        )
    return fig