"""
Tests for dashboard/utils/instancedetail_helpers

Covers:
- build_model_panel
- render_model_panel
"""

import pytest
import plotly
import pandas as pd
from unittest.mock import patch, MagicMock
from xailens.dashboard.utils.instancedetail_helpers import build_model_panel, render_model_panel, ModelPanel

from xailens.exceptions import InvalidConfigurationError
MODULE = "xailens.dashboard.utils.instancedetail_helpers"



@pytest.fixture
def patched_registries():
    with patch(f"{MODULE}.get_xai_adapter") as get_xai, \
         patch(f"{MODULE}.get_viz_adapter") as get_viz:
        xai_class = MagicMock(name="XaiClass")
        xai_class.return_value.get_default_visualisation.return_value = "bar_chart"
        get_xai.return_value = xai_class
        yield get_xai, get_viz


@pytest.fixture
def mock_st():
    with patch(f"{MODULE}.st") as st:
        st.session_state = {}
        yield st


@pytest.fixture
def panel_factory(lr_model):
    def _make(viz_names=("bar_chart",), viz_class=None, explanation="EXPL"):
        xai_instance = MagicMock()
        xai_instance.get_compatible_visualisations.return_value = list(viz_names)
        xai_instance.display_notes = "notes"
        return ModelPanel(
            model=lr_model, xai_class=MagicMock(), xai_instance=xai_instance,
            viz_key="viz-key", viz_name=viz_names[0], viz_class=viz_class,
            explanation=explanation, instance_uuid="inst_0",
        )
    return _make

def test_build_uses_default_visualisation_when_not_in_params(lr_model, patched_registries):
    get_xai, get_viz = patched_registries

    panel = build_model_panel(lr_model, "viz-key", {}, "inst_0")

    get_xai.assert_called_once_with("coefficients")
    assert panel.viz_name == "bar_chart"
    get_viz.assert_called_once_with("bar_chart")
    assert panel.viz_class is get_viz.return_value
    assert panel.instance_uuid == "inst_0"


def test_build_prefers_viz_params_over_default(lr_model, patched_registries):
    _, get_viz = patched_registries

    panel = build_model_panel(lr_model, "viz-key", {"viz-key": "waterfall"}, "inst_0")

    assert panel.viz_name == "waterfall"
    get_viz.assert_called_once_with("waterfall")


def test_build_loads_explanation_for_instance(lr_model, patched_registries):
    get_xai, _ = patched_registries
    xai_class = get_xai.return_value

    panel = build_model_panel(lr_model, "viz-key", {}, "inst_0")

    xai_class.load_local_explanation.assert_called_once_with(lr_model, "inst_0")
    assert panel.explanation is xai_class.load_local_explanation.return_value


@pytest.mark.parametrize("error", [KeyError("does_not_exist"), InvalidConfigurationError("does_not_exist")])
def test_build_sets_viz_class_none_on_lookup_failure(lr_model,
                                                     patched_registries,
                                                     error):
    _, get_viz = patched_registries
    get_viz.side_effect = error

    panel = build_model_panel(lr_model, "viz-key", {}, "inst_0")

    assert panel.viz_class is None


def test_render_seeds_session_state_with_first_option(mock_st,
                                                      panel_factory):
    panel = panel_factory(viz_names=("bar_chart", "waterfall"), viz_class=MagicMock())
    render_model_panel(panel, "plot-1")
    assert mock_st.session_state["viz-key"] == "bar_chart"


def test_render_hides_radio(mock_st, panel_factory):
    render_model_panel(panel_factory(viz_class=MagicMock()), "plot-1")
    mock_st.radio.assert_not_called()


def test_render_shows_radio(mock_st, panel_factory):
    panel = panel_factory(viz_names=("bar_chart", "tabular"), viz_class=MagicMock())
    render_model_panel(panel, "plot-1")
    mock_st.radio.assert_called_once()


def test_render_warns_and_returns_when_viz_unregistered(mock_st, panel_factory):
    panel = panel_factory(viz_class=None)
    render_model_panel(panel, "plot-1")
    mock_st.warning.assert_called_once()
    mock_st.plotly_chart.assert_not_called()


def test_render_warns_when_viz_not_implemented(mock_st, panel_factory):
    viz_class = MagicMock()
    viz_class.return_value.render.side_effect = NotImplementedError
    render_model_panel(panel_factory(viz_class=viz_class), "plot-1")
    mock_st.warning.assert_called_once()

@patch(f"{MODULE}.apply_horizontal_bar_theme", side_effect=lambda fig: fig)
def test_render_dispatches_figure_to_plotly_chart(themed, mock_st, panel_factory):
    fig = plotly.graph_objs.Figure()
    viz_class = MagicMock()
    viz_class.return_value.render.return_value = [fig]

    render_model_panel(panel_factory(viz_class=viz_class), "plot-1")

    themed.assert_called_once_with(fig)
    assert mock_st.plotly_chart.call_args.args[0] is fig
    assert mock_st.plotly_chart.call_args.kwargs["key"] == "plot-1"


@patch(f"{MODULE}.dataframe_to_html_table", return_value="<table/>")
def test_render_dispatches_dataframe_to_html(to_html, mock_st, panel_factory):
    df = pd.DataFrame({"feature": ["age"], "importance": [0.5]})
    viz_class = MagicMock()
    viz_class.return_value.render.return_value = [df]

    render_model_panel(panel_factory(viz_class=viz_class), "plot-1")

    to_html.assert_called_once()
    mock_st.html.assert_any_call("<table/>")


def test_render_dispatches_raw_html(mock_st, panel_factory):
    viz_class = MagicMock()
    viz_class.return_value.render.return_value = ["<b>x</b>"]
    render_model_panel(panel_factory(viz_class=viz_class), "plot-1")
    mock_st.html.assert_any_call("<b>x</b>")