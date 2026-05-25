# xai_compare/dashboard/components/sidebar.py
import streamlit as st
import base64

from pathlib import Path

from xai_compare._version import __version__
from xai_compare.run_explorer import RunExplorer

def sidebar():
    with st.sidebar:
        logo_path = Path(__file__).parent.parent / "assets" / "xai-logo.svg"

        with open(logo_path, "rb") as f:
            svg_base64 = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <div class="logo-container">
                <img src="data:image/svg+xml;base64,{svg_base64}">
            </div>
            """,
            unsafe_allow_html=True
        )

        run_exp = RunExplorer()
        datasets = run_exp.get_datasets()
        st.header("Select:")
        dataset = st.selectbox("Dataset", datasets)

        st.markdown(
            f"""
                    <div class="sidebar-footer">
                        XAI Compare v{__version__}
                    </div>
                    """,
            unsafe_allow_html=True
        )

        return dataset
