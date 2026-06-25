import plotly.express as px
import plotly.graph_objects as go

THEME = {
    "color_sequence": px.colors.qualitative.Safe,
    "font_family": "Inter, sans-serif",
    "axis_title_font_size": 16,
    "axis_title_color": "#333333",
    "axis_tick_font_size": 14,
    "axis_tick_color": "#555555",
    "title_font_size": 18,
    "title_color": "#111111",
    "title_font_weight": "normal",
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
}

def apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        font_family=THEME["font_family"],
        title_font_size=THEME["title_font_size"],
        title_font_color=THEME["title_color"],
        title_font_weight=THEME["title_font_weight"],
        plot_bgcolor=THEME["plot_bgcolor"],
        paper_bgcolor=THEME["paper_bgcolor"],
        xaxis={
            "showgrid": False,
            "title_font": {"size": THEME["axis_title_font_size"],
                           "color":THEME["axis_title_color"]},
            "tickfont": {"size": THEME["axis_tick_font_size"],
                         "color": THEME["axis_tick_color"]}
        },
        yaxis= {
            "range": [0, 1],
            "gridcolor": "#eeeeee",
            "title_font": {"size": THEME["axis_title_font_size"],
                           "color": THEME["axis_title_color"]},
            "tickfont": {"size": THEME["axis_tick_font_size"],
                         "color": THEME["axis_tick_color"]}
        }
    )
    fig.update_layout(showlegend=False)
    return fig