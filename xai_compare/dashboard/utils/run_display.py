import streamlit as st

from xai_compare.run import Run
from xai_compare.dashboard.utils.formatting import format_model_name, format_datetime


def render_run_html(run: Run) -> str:

    model = format_model_name(run)
    xai = ", ".join(run.get("xai_methods") or [])

    return f"""
    <div class="run-cell">
        <table>
            <tr>
                <td class="model" colspan=2><strong>{model}</strong></td>
            </tr>
            <tr>
                 <td class="label">XAI methods</td>
                <td>{xai}</td>
            </tr>
            <tr>
                 <td class="label">Run name</td>
                <td>{run.get('run_name')}</td>
            </tr>
            <tr>
                 <td class="label">Date/time</td>
                <td>{format_datetime(run.datetime)}</td>
            </tr>
        </table></div>
    """


def display_run_cell(col, run: Run):
    col.markdown(render_run_html(run), unsafe_allow_html=True)


def display_comparable_runs(groups: list[list[Run]]):
    if not groups:
        st.info("No comparable models found.")
        return

    h = st.columns([2, 4, 1, 4, 2])
    h[0].markdown("**Dataset**")
    h[1].markdown("**Model A**")
    h[3].markdown("**Model B**")
    st.divider()

    for i, (run_a, run_b) in enumerate(groups):
        cols = st.columns([2, 4, 1, 4, 2])
        cols[0].write(run_a.get("dataset_id"))
        display_run_cell(cols[1], run_a)
        cols[2].markdown("<div style='text-align:center;padding-top:1rem'>vs</div>", unsafe_allow_html=True)
        display_run_cell(cols[3], run_b)
        runs_value = f"{run_a.get('run_uuid')},{run_b.get('run_uuid')}"

        cols[4].markdown(
            f'<a href="/overview?runs={runs_value}" target="_self" class="button">Go to Comparison</a>',
            unsafe_allow_html=True)
        st.divider()