import streamlit as st

from xailens.run import Run
from xailens.dashboard.utils.formatting import format_model_name, format_datetime
from xailens import strings

def _badge_class(model_type: str) -> str:
    t = (model_type or "").upper()
    if t == "LLM":
        return "badge badge-llm"
    if t in ("XGBOOST", "XGB"):
        return "badge badge-xgb"
    if t == "LR":
        return "badge badge-lr"
    return "badge badge-other"


def render_run_cell(run: Run) -> str:
    model_name = format_model_name(run)
    model_type = run.get("model_type", "")
    run_name   = run.get("run_name") or ""
    dt         = format_datetime(run.datetime)
    metrics    = run.get("metrics") or {}
    acc        = metrics.get("accuracy")
    f1         = metrics.get("f1_macro")
    acc_str    = f"accuracy {acc:.2f}" if acc is not None else ""
    f1_str     = f"F1-macro {f1:.2f}"  if f1  is not None else ""
    metric_str = " &middot; ".join(filter(None, [acc_str, f1_str]))

    return f"""
    <div class="run-cell">
        <div class="run-cell-model">
            <span class="{_badge_class(model_type)}">{strings.MODEL_DISPLAY_NAME[model_type]}</span>
            <span class="run-cell-model-name">{model_name}</span>
        </div>
        <div class="run-cell-meta">
            {metric_str}
            &nbsp;&middot;&nbsp; {dt}
        </div>
        <div class="run-cell-run-name">{run_name}</div>
    </div>
    """


def display_comparable_runs(groups: list[list[Run]]):
    if not groups:
        st.info("No comparable models found.")
        return

    rows_html = """
       <div class="run-card-header">
           <span>Dataset</span>
           <span>Model A</span>
           <span></span>
           <span>Model B</span>
           <span></span>
       </div>
       """

    for run_a, run_b in groups:
        runs_value = f"{run_a.get('run_uuid')},{run_b.get('run_uuid')}"
        rows_html += f"""
            <div class="run-card-row">
                <div class="dataset-col">{run_a.get('dataset_id')}</div>
                {render_run_cell(run_a)}
                <div class="vs-col">vs</div>
                {render_run_cell(run_b)}
                <div style="padding-left:12px">
                    <a href="/overview?runs={runs_value}" target="_self" class="button">{strings.BTN_GO_TO_COMPARISON}</a>
                </div>
            </div>
            """

    return rows_html