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
}

def apply_base_theme(fig: go.Figure) -> go.Figure:
    """Applies universal typography and background styling only.
    Chart-specific axis config (range, grid, orientation) stays in the render function."""
    current_title = fig.layout.title.text
    if current_title and not current_title.startswith("<span"):
        updated_title = f"<span style='font-weight: normal;'>{current_title}</span>"
    else:
        updated_title = current_title
    fig.update_layout(
        title={"text": updated_title},
        margin={"t": 40, "b": 10, "l": 10, "r": 10},
        font_family=THEME["font_family"],
        title_font={
            "size": THEME["title_font_size"],
            "color": THEME["title_color"],
            "weight": THEME["title_font_weight"],
        },
        plot_bgcolor=THEME["plot_bgcolor"],
        paper_bgcolor=THEME["paper_bgcolor"],
        showlegend=False,
    )
    fig.update_xaxes(
        title_font={"size": THEME["axis_title_font_size"], "color": THEME["axis_title_color"]},
        tickfont={"size": THEME["axis_tick_font_size"], "color": THEME["axis_tick_color"]},
    )
    fig.update_yaxes(
        title_font={"size": THEME["axis_title_font_size"], "color": THEME["axis_title_color"]},
        tickfont={"size": THEME["axis_tick_font_size"], "color": THEME["axis_tick_color"]},
    )
    return fig

def apply_theme(fig: go.Figure) -> go.Figure:
    fig = apply_base_theme(fig)
    fig.update_layout(
        xaxis={"showgrid": False},
        yaxis={"range": [0, 1], "gridcolor": "#eeeeee"},
    )
    return fig


def apply_horizontal_bar_theme(fig: go.Figure, xaxis_title: str = "") -> go.Figure:
    fig = apply_base_theme(fig)
    fig.update_layout(
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
    return fig
