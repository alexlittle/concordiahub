
import streamlit as st

from xai_compare.dashboard.components.layout import apply_layout
from xai_compare.dashboard.services import get_explorer
from xai_compare.dashboard.components.formatting import format_model_name, format_datetime
from xai_compare.services.comparison import build_comparison_df
from xai_compare import constants

# Get/load the necessary data for this page
explorer = get_explorer()

runs_param = st.query_params.get("runs")
if not runs_param:
    st.warning("No runs selected")
    st.stop()
run_ids = runs_param.split(",")

runs = explorer.get_runs_by_uuids(run_ids)

if len(runs) != 2:
    st.error("You need exactly 2 runs to compare")
    st.stop()

run_a, run_b = runs


# Page layout/display
apply_layout("XAI Compare")

st.subheader(f"**Dataset:** {run_a.get('dataset_id')}")
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"**Model:** {format_model_name(run_a)}")
    st.write(f"**Date:** {format_datetime(run_a.get('datetime'))}")
    st.write(f"**XAI Methods:** {run_a.get('xai_methods')}")
    st.write("**Classification Report:**")
    st.dataframe(run_a.classification_report())

with col2:
    st.subheader(f"**Model:** {format_model_name(run_b)}")
    st.write(f"**Date:** {format_datetime(run_b.get('datetime'))}")
    st.write(f"**XAI Methods:** {run_b.get('xai_methods')}")
    st.write("**Classification Report:**")
    st.dataframe(run_b.classification_report())

df = build_comparison_df(run_a=run_a, run_b=run_b)



rows_html = ""
for _, row in df.iterrows():
    sample_uuid = row[run_a.get(constants.DATA_SCHEMA_ID_COLUMN)]
    rows_html += f"""
        <tr>
            <td><a href='/detail?runs={runs_param}&sample_id={sample_uuid}' target='_self'>{sample_uuid}</a></td>
            <td>{row['Comparison']}</td>
            <td class="report-table-align">{row['True Value']}</td>
            <td class="report-table-align">{row[f"{format_model_name(run_a)} prediction"]}</td>
            <td class="report-table-align">{row[f"{format_model_name(run_b)} prediction"]}</td>
        </tr>
    """

table_html = f"""
    <table class="report-table">
        <thead>
            <tr>
                <th>sample_uuid</th>
                <th>Comparison</th>
                <th>True Value</th>
                <th>{format_model_name(run_a)}</th>
                <th>{format_model_name(run_b)}</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
"""

st.html(table_html)










