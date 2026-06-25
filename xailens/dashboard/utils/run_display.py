import streamlit as st

from xailens.entities.modeleval import ModelEval
from xailens import strings
from xailens.dashboard.utils.styles import get_badge_class

def render_modeleval_cell(modeleval: ModelEval) -> str:
    model_display_name = modeleval.get("model_display_name")
    model_type = modeleval.get("model_type", "")
    metrics = modeleval.get("metrics") or {}
    acc = metrics.get("accuracy")
    f1 = metrics.get("f1_macro")
    acc_str = f"accuracy {acc:.2f}" if acc is not None else ""
    f1_str = f"F1-macro {f1:.2f}"  if f1  is not None else ""
    metric_str = " &middot; ".join(filter(None, [acc_str, f1_str]))

    return f"""
    <div class="run-cell">
        <div class="run-cell-model">
            <span class="{get_badge_class(model_type)}">{strings.MODEL_DISPLAY_NAME[model_type]}</span>
            <span class="run-cell-model-name">{model_display_name}</span>
        </div>
        <div class="run-cell-meta">
            {metric_str}
        </div>
        <div class="run-cell-run-name">{model_display_name}</div>
    </div>
    """


def display_comparable_runs(groups: list[list[ModelEval]]):
    if not groups:
        st.info("No comparable models found.")
        return ""

    rows_html = """
       <div class="run-card-header">
           <span>Dataset</span>
           <span>Model A</span>
           <span></span>
           <span>Model B</span>
           <span></span>
       </div>
       """

    for modeleval_a, modeleval_b in groups:
        modelevals_value = f"{modeleval_a.get('model_uuid')},{modeleval_b.get('model_uuid')}"
        rows_html += f"""
            <div class="run-card-row">
                <div class="dataset-col">{modeleval_a.get('dataset_id')}</div>
                {render_modeleval_cell(modeleval_a)}
                <div class="vs-col">vs</div>
                {render_modeleval_cell(modeleval_b)}
                <div style="padding-left:12px">
                    <a href="/?modelevals={modelevals_value}" target="_self" class="button">{strings.BTN_GO_TO_COMPARISON}</a>
                </div>
            </div>
            """

    return rows_html