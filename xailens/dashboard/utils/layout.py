
import streamlit as st

from xailens.dashboard.utils.styles import load_css
from xailens import strings
from xailens.dashboard.utils.render import sidebar


def apply_layout(title: str,
                 all_model_names: list = None,
                 selected_model_names: list = None,
                 dataset_id: str = None):
    # Page title
    st.set_page_config(page_title=f"{strings.APP_NAME} - {title}", layout="wide")

    # inject css
    load_css()

    # Sidebar
    with st.sidebar:
        st.html(sidebar.render_header())
        if all_model_names:
            st.html(f"<h3>{strings.SIDEBAR_MODEL_FILTER_TITLE}</h3>")
            selected = []
            for name in sorted(all_model_names):
                if st.checkbox(name, value=name in selected_model_names, key=f"chk_{dataset_id}_{name}"):
                    selected.append(name)

            if st.button("Apply selection"):
                st.query_params["dataset"] = dataset_id
                st.query_params["models"] = ",".join(selected)
                st.rerun()

        st.html(sidebar.render_footer())

    # Page title
    st.html(f"<h1 class='header'>{title}</h1>")



