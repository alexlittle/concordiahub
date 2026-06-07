import streamlit as st
from xai_compare import constants
from xai_compare.dashboard.utils.formatting import format_model_name, format_datetime



def render_run_summary(run):
    st.subheader(f"Model: {format_model_name(run)}")
    st.markdown(f"**Date**: {format_datetime(run.get('datetime'))}")
    st.markdown(f"**XAI Methods**: {run.get('xai_methods')}")
    st.markdown("**Classification Report:**")
    st.dataframe(run.classification_report())


def generate_comparison_rows_html(df, run_a, run_b):
    rows_html = ""
    for _, row in df.iterrows():
        sample_id = row[run_a.get(constants.DATA_SCHEMA_ID_COLUMN)]

        if row['True Value'] == row[f"{format_model_name(run_a)} prediction"]:
            run_a_icon = "<span class='tick'>&#10004;</span>"
        else:
            run_a_icon = "<span class='cross'>&#10008;</span>"

        if row['True Value'] == row[f"{format_model_name(run_b)} prediction"]:
            run_b_icon = "<span class='tick'>&#10004;</span>"
        else:
            run_b_icon = "<span class='cross'>&#10008;</span>"

        rows_html += f"""
        <tr>
            <td><a href='/detail?runs={run_a.get("run_uuid")},{run_b.get("run_uuid")}&sample_id={sample_id}' target='_self'>{sample_id}</a></td>
            <td>{row['Comparison']}</td>
            <td class="report-table-align">{row['True Value']}</td>
            <td class="report-table-align">
                {row[f"{format_model_name(run_a)} prediction"]}
                {run_a_icon} 
            </td>
            <td class="report-table-align">
                {row[f"{format_model_name(run_b)} prediction"]}
                {run_b_icon} 
            </td>
        </tr>
        """
    return rows_html


def generate_comparison_table_html(df, run_a, run_b):
    rows_html = generate_comparison_rows_html(df, run_a, run_b)

    return f"""
    <table class="report-table">
        <thead>
            <tr>
                <th>Sample Identifier</th>
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
