import numpy as np
import streamlit as st
from xai_compare.dashboard.components.formatting import format_model_name

from xai_compare import constants

def ensure_sample_index(df, key="sample_uuid"):
    # already a column so set as index
    if key in df.columns:
        return df.set_index(key)

    # already index with correct name
    if df.index.name == key:
        return df

    # index exists but unnamed so assume it's sample_uuid
    if df.index.name is None:
        df = df.copy()
        df.index.name = key
        return df

    # totally missing so fail
    raise ValueError(
        f"{key} not found in columns or index. "
        f"Columns: {df.columns}, Index: {df.index.name}"
    )



def build_comparison_df(run_a,
                        run_b,
                        ensure_sample_index_fn=ensure_sample_index):
    name_a = format_model_name(run_a)
    name_b = format_model_name(run_b)

    if name_a == name_b:
        raise ValueError(f"Cannot compare runs with identical model names: '{name_a}'")

    y_true_a, y_pred_a = run_a.load_targets()
    _, y_pred_b = run_b.load_targets()

    y_true_a = ensure_sample_index_fn(
        y_true_a,run_a.get(constants.DATA_SCHEMA_ID_COLUMN))
    y_pred_a = ensure_sample_index_fn(
        y_pred_a, run_a.get(constants.DATA_SCHEMA_ID_COLUMN))
    y_pred_b = ensure_sample_index_fn(
        y_pred_b, run_b.get(constants.DATA_SCHEMA_ID_COLUMN))

    y_true_a = y_true_a.rename(columns=
        {run_a.get(constants.DATA_SCHEMA_TARGET_COLUMN): constants.TRUE_VALUE_COLUMN_LABEL})
    y_pred_a.columns = [f"{name_a} {constants.PREDICTION_COLUMN_LABEL}"]
    y_pred_b.columns = [f"{name_b} {constants.PREDICTION_COLUMN_LABEL}"]

    df = y_true_a.join(y_pred_a).join(y_pred_b)

    # relabel columns and show which are correct/wrong
    pred_a = f"{name_a} {constants.PREDICTION_COLUMN_LABEL}"
    pred_b = f"{name_b} {constants.PREDICTION_COLUMN_LABEL}"

    p1_correct = df[pred_a] == df[constants.TRUE_VALUE_COLUMN_LABEL]
    p2_correct = df[pred_b] == df[constants.TRUE_VALUE_COLUMN_LABEL]
    same_pred = df[pred_a] == df[pred_b]

    COMPARISON_ORDER = {
        constants.COMPARISON_BOTH_INCORRECT_DIFF: 1,
        constants.COMPARISON_BOTH_INCORRECT_SAME: 2,
        f"{name_a} {constants.COMPARISON_CORRECT}": 3,
        f"{name_b} {constants.COMPARISON_CORRECT}": 4,
        constants.COMPARISON_BOTH_CORRECT: 5,
    }

    df["Comparison"] = np.select(
        [
            p1_correct & p2_correct,
            p1_correct & ~p2_correct,
            ~p1_correct & p2_correct,
            ~p1_correct & ~p2_correct & same_pred,
            ~p1_correct & ~p2_correct & ~same_pred,

        ],
        [
            constants.COMPARISON_BOTH_CORRECT,
            f"{name_a} {constants.COMPARISON_CORRECT}",
            f"{name_b} {constants.COMPARISON_CORRECT}",
            constants.COMPARISON_BOTH_INCORRECT_SAME,
            constants.COMPARISON_BOTH_INCORRECT_DIFF
        ],
        default="Unknown"
    )

    col = df.pop("Comparison")
    df.insert(0, "Comparison", col)

    df["comparison_order"] = df["Comparison"].map(COMPARISON_ORDER)
    df = df.sort_values(by=["comparison_order",
        constants.TRUE_VALUE_COLUMN_LABEL]).drop(columns=["comparison_order"])

    return df.reset_index()

