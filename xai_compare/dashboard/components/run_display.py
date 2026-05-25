import streamlit as st

from datetime import datetime


def format_datetime(dt_str: str) -> str:
    dt = datetime.strptime(dt_str, "%Y%m%d-%H%M%S")
    return dt.strftime("%H:%M %d %b %Y")

def render_run_html(run) -> str:
    subtype = run.get("model_subtype")
    model = run["model_type"] + (f" ({subtype})" if subtype else "")
    xai = ", ".join(run.get("xai_methods") or [])

    return f"""
    <div class="run-cell">
        <table>
            <tr>
                <td class="label">Model</td>
                <td><code>{model}</code></td>
            </tr>
            <tr>
                 <td class="label">XAI methods</td>
                <td><code>{xai}</code></td>
            </tr>
            <tr>
                 <td class="label">Run name</td>
                <td><code>{run['run_name']}</code></td>
            </tr>
            <tr>
                 <td class="label">Date/time</td>
                <td><code>{format_datetime(run['datetime'])}</code></td>
            </tr>
        </table></div>
    """


def display_run_cell(col, run):
    col.markdown(render_run_html(run), unsafe_allow_html=True)

def display_comparable_runs(groups: list[list[dict]]):
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
        cols[0].write(run_a["dataset_id"])
        display_run_cell(cols[1], run_a)
        cols[2].markdown("<div style='text-align:center;padding-top:1rem'>vs</div>", unsafe_allow_html=True)
        display_run_cell(cols[3], run_b)
        runs_value = f"{run_a['run_uuid']},{run_b['run_uuid']}"

        cols[4].markdown(
            f'<a href="/overview?runs={runs_value}" target="_self">Compare</a>',
            unsafe_allow_html=True)


    st.divider()