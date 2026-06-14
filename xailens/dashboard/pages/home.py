import streamlit as st

from xailens.dashboard.utils.layout import apply_layout
from xailens.dashboard.utils.run_display import display_comparable_runs
from xailens.dashboard.utils.filter import build_filter_pills_html, apply_dataset_filter, apply_model_filter
from xailens.dashboard.services import get_explorer

from xailens import strings

def main():
    explorer = get_explorer()

    all_runs = explorer.get_comparable_runs()

    apply_layout(strings.HOMEPAGE_TITLE)

    st.html(strings.HOMEPAGE_HELPER)


    active_dataset = st.query_params.get("dataset_filter", "all")
    active_models = st.query_params.get("models_filter", "all")

    current_params = {
        "models_filter": active_models,
        "dataset_filter": active_dataset,
    }
    model_pill_options = explorer.get_model_filter_options()
    dataset_pill_options = explorer.get_dataset_filter_options()

    st.html(build_filter_pills_html(active=active_dataset,
                                    param="dataset_filter",
                                    pill_options=dataset_pill_options,
                                    variant="dataset",
                                    current_params=current_params))

    st.html(build_filter_pills_html(active=active_models,
                                    param="models_filter",
                                    pill_options=model_pill_options,
                                    variant="model",
                                    current_params=current_params))

    filtered_runs = apply_model_filter(all_runs, active_models)
    filtered_runs = apply_dataset_filter(filtered_runs, active_dataset)

    runs = display_comparable_runs(filtered_runs)
    if runs is not None:
        st.html(runs)

if __name__ == "__main__":
    main()
