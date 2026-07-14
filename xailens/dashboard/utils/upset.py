"""
Function to build an UpSet matrix-set plot
"""


import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from xailens.config import PLOT_UPSET_CORRECT_COLOR, PLOT_UPSET_INCORRECT_COLOR


def build_upset_figure(combo_df: pd.DataFrame,
                       set_sizes: pd.Series,
                       max_combos: int = 25) -> go.Figure:
    """
    Builds the upset matrix-set plot

    Args:
        combo_df (pd.DataFrame): dataframe containing model combinations
        set_sizes (pd.Series): set sizes for each model
        max_combos (int): maximum number of combinations to display

    Returns:
        go.Figure: upset matrix-set plot
    """

    model_cols = list(set_sizes.index)
    combo_df = combo_df.head(max_combos).reset_index(drop=True)
    n_combos = len(combo_df)
    n_models = len(model_cols)

    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.65, 0.35],
        column_widths=[0.20, 0.80],
        horizontal_spacing=0.02,
        vertical_spacing=0.03,
        specs=[
            [None, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
    )

    # --- Top-right: combo size bar chart ---
    fig.add_trace(
        go.Bar(
            x=list(range(n_combos)),
            y=combo_df["count"],
            marker_color=PLOT_UPSET_INCORRECT_COLOR,
            hovertext=[
                f"{c} instances incorrect in models: <br>" + ", ".join(
                    m for m in model_cols if combo_df.loc[i, m] == 1
                )
                for i, c in enumerate(combo_df["count"])
            ],
            hoverinfo="text",
            showlegend=False,
        ),
        row=1, col=2,
    )

    # --- Bottom-left: set size bar chart ---
    fig.add_trace(
        go.Bar(
            x=set_sizes.values[::-1],
            y=list(range(n_models)),
            orientation="h",
            marker_color=PLOT_UPSET_INCORRECT_COLOR,
            name="",
            hovertemplate="%{x} incorrect<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # --- Bottom-right: full matrix, every dot coloured correct/incorrect ---
    correct_x, correct_y = [], []
    incorrect_x, incorrect_y = [], []
    row_order = model_cols[::-1]
    combo_key_list = [
        ",".join(m for m in model_cols if combo_df.loc[i, m] == 1)
        for i in range(n_combos)
    ]

    correct_cd, incorrect_cd = [], []
    for i in range(n_combos):
        for j, m in enumerate(row_order):
            if combo_df.loc[i, m] == 1:
                incorrect_x.append(i)
                incorrect_y.append(j)
                incorrect_cd.append(combo_key_list[i])
            else:
                correct_x.append(i)
                correct_y.append(j)
                correct_cd.append(combo_key_list[i])

    # base layer: every "correct" dot
    fig.add_trace(
        go.Scatter(
            x=correct_x, y=correct_y, mode="markers",
            marker={"size": 14, "color": PLOT_UPSET_CORRECT_COLOR},
            customdata=correct_cd,
            name="Correct",
            hovertemplate="Models: %{customdata}<br>Click to view instances<extra></extra>",
            showlegend=True,
        ),
        row=2, col=2,
    )

    # connecting lines + incorrect dots, per combo
    for i in range(n_combos):
        active_js = [j for j, m in enumerate(row_order) if combo_df.loc[i, m] == 1]
        if not active_js:
            continue
        fig.add_trace(
            go.Scatter(
                x=[i, i], y=[min(active_js), max(active_js)],
                mode="lines",
                line={"color": PLOT_UPSET_INCORRECT_COLOR, "width": 2},
                name="Incorrect",
                hovertemplate="Models: %{customdata}<br>Click to view instances<extra></extra>",
                showlegend=False,
            ),
            row=2, col=2,
        )

    fig.add_trace(
        go.Scatter(
            x=incorrect_x, y=incorrect_y, mode="markers",
            marker={"size": 14, "color": PLOT_UPSET_INCORRECT_COLOR},
            customdata=incorrect_cd,
            name="Incorrect",
            hovertemplate="Models: %{customdata}<br>Click to view instances<extra></extra>",
            showlegend=True,
        ),
        row=2, col=2,
    )

    # --- Axis formatting ---
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(autorange="reversed", title_text="Set size", row=2, col=1)
    fig.update_yaxes(
        tickmode="array", tickvals=list(range(n_models)),
        ticktext=model_cols[::-1], row=2, col=1,
    )
    fig.update_xaxes(showticklabels=False, row=2, col=2)
    fig.update_yaxes(
        tickmode="array", tickvals=list(range(n_models)),
        ticktext=[""] * n_models, showgrid=False, row=2, col=2,
    )

    fig.update_xaxes(range=[-0.5, n_combos - 0.5], row=1, col=2)
    fig.update_xaxes(range=[-0.5, n_combos - 0.5], matches="x3", row=2, col=2)

    fig.update_layout(
        height=500,
        bargap=0.3,
        plot_bgcolor="white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return fig