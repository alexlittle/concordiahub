
import streamlit as st

from xailens.dashboard.utils.styles import load_css
from xailens import strings, constants
from xailens.dashboard.utils import sidebar


def apply_layout(title: str,
                 dataset_id: str = None,
                 all_model_names: list = None,
                 selected_model_names: list = None,
                 selected_instance_group: str = None,):
    # Page title
    st.set_page_config(page_title=f"{strings.APP_NAME} - {title}", layout="wide")

    # inject css
    load_css()

    # Sidebar
    with st.sidebar:
        st.html(sidebar.render_header())
        if all_model_names:
            st.html(f"<h3>{strings.SIDEBAR_MODEL_FILTER_TITLE}</h3>")
            selected_ds = []
            for name in sorted(all_model_names):
                if st.checkbox(name, value=name in selected_model_names, key=f"chk_{dataset_id}_{name}"):
                    selected_ds.append(name)

            selected_ig = None
            if selected_instance_group:
                options = list(constants.INSTANCE_GROUPS)
                selected_ig = st.radio(
                    label=f"{strings.SIDEBAR_INSTANCE_GROUP_FILTER_TITLE}",
                    options=options,
                    format_func=lambda k: constants.INSTANCE_GROUPS[k],
                    index=options.index(selected_instance_group) if selected_instance_group in options else 0,
                    key=f"radio_{dataset_id}_instance_group"
                )

            if st.button("Apply selection"):
                st.query_params["dataset"] = dataset_id
                st.query_params["models"] = ",".join(selected_ds)
                if selected_ig:
                    st.query_params["instances"] = selected_ig
                st.rerun()

        st.html(sidebar.render_footer())

    # Page title
    st.html(f"<h1 class='header'>{title}</h1>")



