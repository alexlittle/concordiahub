"""
Functions to build and render an UpSet-style matrix-set plot
"""


import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plotly.subplots import make_subplots

from xailens.entities.dataset import Dataset
from xailens.config import PLOTLY_CONFIG, PLOT_UPSET_CORRECT_COLOR, PLOT_UPSET_INCORRECT_COLOR
from xailens.dashboard.utils.plotly_theme import apply_upset_theme

from xailens import strings

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

    # Make the 4 different areas for the plots:
    # top-left - empty
    # top-right bar chart of num instances in the set
    # bottom-left - bar chart of num instances wrong per model
    # bottom-right - matrix set dots with the groups of misclassifications
    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.65, 0.35],
        column_widths=[0.20, 0.80],
        horizontal_spacing=0.02,
        vertical_spacing=0.03,
        specs=[
            [None, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
    )

    # Overall title to the set of subplots
    fig.update_layout(
        title_text = strings.MATRIX_SET_PLOT_TITLE,
        title_x=0,
        title_xanchor="left",
    )

    correct_x, correct_y = [], []
    incorrect_x, incorrect_y = [], []
    row_order = model_cols[::-1]
    combo_key_list = [
        ",".join(m for m in model_cols if combo_df.loc[i, m] == 1)
        for i in range(n_combos)
    ]

    #  Top-right -  bar chart of the set combinations
    fig.add_trace(
        go.Bar(
            x=list(range(n_combos)),
            y=combo_df["count"],
            marker_color=PLOT_UPSET_INCORRECT_COLOR,
            customdata=combo_key_list,
            hovertext=[
                f"{c} instances incorrect in models: <br>" + ", ".join(
                    m for m in model_cols if combo_df.loc[i, m] == 1
                ) + "<br>Click to view instances"
                for i, c in enumerate(combo_df["count"])
            ],
            hoverinfo="text",
            showlegend=False,
            selected={"marker": {"opacity": 1}},
            unselected={"marker": {"opacity": 1}},
        ),
        row=1, col=2,
    )

    # Bottom-left: bar chart of the number of misclassifications per model
    fig.add_trace(
        go.Bar(
            x=set_sizes.values[::-1],
            y=list(range(n_models)),
            orientation="h",
            marker_color=PLOT_UPSET_INCORRECT_COLOR,
            name="",
            hovertemplate="%{x} instances incorrect<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # Bottom-right - full matrix, every dot coloured for correct/incorrect


    # calculate what combination dots should be shown as correct/incorrect
    incorrect_cd = []
    for i in range(n_combos):
        for j, m in enumerate(row_order):
            if combo_df.loc[i, m] == 1:
                incorrect_x.append(i)
                incorrect_y.append(j)
                incorrect_cd.append(combo_key_list[i])
            else:
                correct_x.append(i)
                correct_y.append(j)

    # base - every "correct" dot
    fig.add_trace(
        go.Scatter(
            x=correct_x,
            y=correct_y,
            mode="markers",
            marker={"size": 14, "color": PLOT_UPSET_CORRECT_COLOR},
            name="Correct",
            hoverinfo="skip",
            showlegend=True,
            selected={"marker": {"opacity": 1}},
            unselected={"marker": {"opacity": 1}},
        ),
        row=2, col=2,
    )

    # connecting lines and incorrect dots
    for i in range(n_combos):
        active_js = [j for j, m in enumerate(row_order) if combo_df.loc[i, m] == 1]
        if not active_js:
            continue
        fig.add_trace(
            go.Scatter(
                x=[i, i],
                y=[min(active_js), max(active_js)],
                mode="lines",
                line={"color": PLOT_UPSET_INCORRECT_COLOR, "width": 2},
                name="Incorrect",
                showlegend=False,
            ),
            row=2, col=2,
        )

    fig.add_trace(
        go.Scatter(
            x=incorrect_x,
            y=incorrect_y,
            mode="markers",
            marker={"size": 14, "color": PLOT_UPSET_INCORRECT_COLOR},
            customdata=incorrect_cd,
            name="Incorrect",
            hovertemplate="Models: %{customdata}<br>Click to view instances<extra></extra>",
            showlegend=True,
            selected={"marker": {"opacity": 1}},
            unselected={"marker": {"opacity": 1}},
        ),
        row=2, col=2,
    )

    # --- Axis formatting ---
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text="Instances incorrect", row=1, col=2)
    fig.update_xaxes(autorange="reversed", title_text="Instances incorrect", row=2, col=1)
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
        legend={"orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1},
    )

    return fig

def render_upset_figure(dataset: Dataset,
                        navigate_on_click: bool = True) -> None:
    """
    Renders a matrix-set plotly figure

    Args:
        dataset (Dataset): dataset to base the figure on
        navigate_on_click (bool): whether to actually force a whole page rerun.
            When this figure is on the page that's linked to, this should
            be set to False. Defaults to True

    """
    combo_df, set_sizes = dataset.get_upset_data()
    fig = build_upset_figure(combo_df, set_sizes)
    fig = apply_upset_theme(fig, show_title=True)

    # Display the chart and catch events (clicks) on the chart
    event = st.plotly_chart(
        fig,
        width='stretch',
        config=PLOTLY_CONFIG,
        on_select="rerun",
        selection_mode="points",
        key="upset_matrix",
    )

    # when clicked on the points direct to correct page/set
    if event and event["selection"]["points"]:
        point = event["selection"]["points"][0]
        set_key = point.get("customdata")
        if set_key and set_key != st.query_params.get("matrix_set"):
            merged_params = dict(st.query_params)
            merged_params["matrix_set"] = set_key
            if navigate_on_click:
                st.switch_page("pages/instances.py",
                               query_params=merged_params)
            else:
                st.query_params["matrix_set"] = set_key