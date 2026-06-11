# xailens/dashboard/components/sidebar.py
import streamlit as st
import base64
import urllib.parse

from pathlib import Path

from xailens._version import __version__
from xailens.dashboard.services import get_explorer
from xailens import strings

def sidebar():
    with st.sidebar:
        logo_path = Path(__file__).parent.parent / "assets" / "xai-lens-logo.svg"

        with open(logo_path, "rb") as f:
            svg_base64 = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <a href="/" style="text-decoration: none;" target="_self">
                <div class="logo-container">
                    <img src="data:image/svg+xml;base64,{svg_base64}">
                </div>
                <h1 class="logo">{strings.APP_NAME}</h1>
            </a>
            """,
            unsafe_allow_html=True
        )

        explorer = get_explorer()
        datasets = explorer.get_datasets()

        st.html(f"<h2>{strings.SIDEBAR_FILTER_TITLE}</h2>")
        dataset = st.selectbox("Dataset", datasets)

        models = explorer.get_models_for_dataset(dataset)
        model_names = list(models.keys())

        model_a = st.selectbox("Model A", model_names, disabled=len(model_names) == 0)
        model_b = st.selectbox("Model B", model_names, disabled=len(model_names) == 0)

        if model_a and model_b and model_a != model_b:
            params = urllib.parse.urlencode({
                "runs": f"{models[model_a]},{models[model_b]}"
            })
            st.markdown(
                f'<a href="/overview?{params}" target="_self" class="button">'
                'Go to Comparison</a>',
                unsafe_allow_html=True
            )
        else:
            if model_a == model_b:
                st.caption("Please select two different models.")

        st.markdown(
            f"""
                    <div class="sidebar-footer">
                        XAI Compare v{__version__}
                    </div>
                    """,
            unsafe_allow_html=True
        )

        return dataset
