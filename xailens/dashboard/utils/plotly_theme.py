import plotly.express as px
import plotly.graph_objects as go

THEME = {
    "color_sequence": px.colors.qualitative.Safe,
    "font_family": "Inter, sans-serif",
    "axis_title_font_size": 16,
    "axis_title_color": "#333333",
    "axis_tick_font_size": 14,
    "axis_tick_color": "#555555",
    "title_font_size": 20,
    "title_color": "#111111",
    "title_font_weight": "normal",
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
    "hover_font_size": 14,
    "hover_font_color": "#333333",
    "hover_bgcolor": "white",
    "hover_bordercolor": "#cccccc",
    "highlight_color": "#555555",
}

def apply_base_theme(fig: go.Figure, show_title: bool=True) -> go.Figure:

    layout_updates = {
        "margin": {"t": 40, "b": 10, "l": 10, "r": 10},
        "font_family": THEME["font_family"],
        "plot_bgcolor": THEME["plot_bgcolor"],
        "paper_bgcolor": THEME["paper_bgcolor"],
        "showlegend": False,
    }
    if not show_title:
        fig.layout.title = None
    else:
        current_title = fig.layout.title.text if fig.layout.title else None
        if current_title and not current_title.startswith("<span"):
            updated_title = f"<span style='font-weight: normal;'>{current_title}</span>"
        else:
            updated_title = current_title

        layout_updates["title"] = {"text": updated_title}
        layout_updates["title_font"] = {
            "size": THEME["title_font_size"],
            "color": THEME["title_color"],
            "weight": THEME["title_font_weight"],
        }

    # Apply everything
    fig.update_layout(**layout_updates)
    fig.update_xaxes(
        title_font={"size": THEME["axis_title_font_size"], "color": THEME["axis_title_color"]},
        tickfont={"size": THEME["axis_tick_font_size"], "color": THEME["axis_tick_color"]},
    )
    fig.update_yaxes(
        title_font={"size": THEME["axis_title_font_size"], "color": THEME["axis_title_color"]},
        tickfont={"size": THEME["axis_tick_font_size"], "color": THEME["axis_tick_color"]},
    )
    return fig

def apply_theme(fig: go.Figure, show_title: bool=True) -> go.Figure:
    fig = apply_base_theme(fig, show_title=show_title)
    fig.update_layout(
        hoverlabel={
            "font_family": THEME["font_family"],
            "font_size": 14,
            "font_color": THEME["hover_font_color"],
            "bgcolor": "white",
            "bordercolor": "#cccccc",
        },
        xaxis={"showgrid": False},
        yaxis={"range": [0, 1], "gridcolor": "#eeeeee"},
    )
    return fig


def apply_horizontal_bar_theme(fig: go.Figure, xaxis_title: str = "") -> go.Figure:
    fig = apply_base_theme(fig)
    fig.update_layout(
        hoverlabel={
            "font_family": THEME["font_family"],
            "font_size": 14,
            "font_color": THEME["hover_font_color"],
            "bgcolor": "white",
            "bordercolor": "#cccccc",
        },
        xaxis={
            "title": xaxis_title,
            "showgrid": True,
            "gridcolor": "#eeeeee",
            "zeroline": True,
            "zerolinewidth": 2,
            "zerolinecolor": "black",
        },
        yaxis={"showgrid": False},
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>"
    )
    return fig
