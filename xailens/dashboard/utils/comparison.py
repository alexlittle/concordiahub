import streamlit as st
from xailens import constants
from xailens.formatting import format_model_name, format_datetime



def render_run_summary(run):
    st.subheader(f"Model: {format_model_name(run)}")
    st.markdown(f"**Date**: {format_datetime(run.get('datetime'))}")
    st.markdown(f"**XAI Methods**: {run.get('xai_methods')}")

    metrics = run.get("metrics") or {}

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{metrics.get("accuracy"):.1%}")
    col2.metric("Macro F1", f"{metrics.get("f1_macro"):.3f}")

    col3.metric("Weighted F1", f"{metrics.get("f1_weighted"):.3f}")

    n_failed = metrics['n_failed']
    n_samples = metrics['n_samples']

    if n_failed > 0:
        col4.metric("Failed samples", n_failed,
                    delta=f"{n_failed / n_samples:.0%} of run",
                    delta_color="inverse")
    else:
        col4.metric("Samples", n_samples)

    with st.expander("Full classification report"):
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


