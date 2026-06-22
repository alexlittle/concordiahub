import streamlit as st
from xailens import constants, strings
from xailens.formatting import format_model_name, format_datetime
from xailens.dashboard.utils.styles import get_badge_class


def render_run_summary(run):
    badge_class = get_badge_class(run.get("model_type"))
    badge_label = run.get("model_type")
    run_name = run.get("run_name")
    model_name = format_model_name(run)
    date_str = format_datetime(run.get("datetime"))
    dataset_id = run.get("dataset_id")
    methods_html = "".join(
        f'<span class="tag">{strings.XAI_METHOD_DISPLAY_NAME[m]}</span> ' for m in (run.get("xai_methods") or [])
    )

    st.html(f"""
    <div class="model-header">
        <span class="{badge_class}">{strings.MODEL_DISPLAY_NAME[badge_label]}</span>
        <h3>{model_name}</h3>
    </div>
    <p class="model-meta">
        {date_str}
        &nbsp;&middot;&nbsp;
        <span class="tag">{dataset_id}</span>
        {methods_html}
    </p>
    <p class="model-meta">
        <em>Run name</em>: {run_name}<br/>
        <em>Run ID</em>: {run.get("runmodel_uuid_uuid")}
    </p>
    """)

    metrics = run.get("metrics") or {}
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics.get('accuracy'):.1%}")
    col2.metric("Macro F1", f"{metrics.get('f1_macro'):.3f}")
    col3.metric("Weighted F1", f"{metrics.get('f1_weighted'):.3f}")

    n_failed = metrics['n_failed']
    n_instances = metrics['n_instances']
    if n_failed > 0:
        col4.metric("Failed instances", n_failed,
                    delta=f"{n_failed / n_instances:.0%} of run",
                    delta_color="inverse")
    else:
        col4.metric("Instances", n_instances)

    with st.expander("Full classification report"):
        st.dataframe(run.classification_report())



def generate_comparison_rows_html(df, modeleval_a, modeleval_b):
    rows_html = ""
    for _, row in df.iterrows():
        instance_id = row[modeleval_a.get(constants.DATA_SCHEMA_ID_COLUMN)]

        if row['True Value'] == row[f"{modeleval_a.get('model_display_name')} prediction"]:
            modeleval_a_icon = "<span class='tick'>&#10004;</span>"
        else:
            modeleval_a_icon = "<span class='cross'>&#10008;</span>"

        if row['True Value'] == row[f"{modeleval_b.get('model_display_name')} prediction"]:
            modeleval_b_icon = "<span class='tick'>&#10004;</span>"
        else:
            modeleval_b_icon = "<span class='cross'>&#10008;</span>"

        rows_html += f"""
        <tr>
            <td><a href='/pwinstance?modelevals={modeleval_a.get("model_uuid")},{modeleval_b.get("model_uuid")}&instance_id={instance_id}' target='_self'>{instance_id}</a></td>
            <td>{row['Comparison']}</td>
            <td class="report-table-align">{row['True Value']}</td>
            <td class="report-table-align">
                {row[f"{modeleval_a.get('model_display_name')} prediction"]}
                {modeleval_a_icon} 
            </td>
            <td class="report-table-align">
                {row[f"{modeleval_b.get('model_display_name')} prediction"]}
                {modeleval_b_icon} 
            </td>
            <td></td>
        </tr>
        """
    return rows_html


def generate_comparison_table_html(df, modeleval_a, modeleval_b):
    rows_html = generate_comparison_rows_html(df, modeleval_a, modeleval_b)

    return f"""
    <table class="report-table">
        <thead>
            <tr>
                <th>Instance Identifier</th>
                <th>Comparison</th>
                <th>True Value</th>
                <th>{modeleval_a.get("model_display_name")}</th>
                <th>{modeleval_b.get("model_display_name")}</th>
                <th>Spearman Rank Correlation (&rho;)</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """


