import numpy as np
from xai_compare.dashboard.components.formatting import format_model_name


DATA_SCHEMA_ID_COLUMN = 'data_schema.sample_id_column'
DATA_SCHEMA_TARGET_COLUMN = 'data_schema.target_column'

PREDICTION_COLUMN_LABEL = "prediction"
TRUE_VALUE_COLUMN_LABEL = "True Value"

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



def build_comparison_df(run_a, run_b, ensure_sample_index_fn=ensure_sample_index):
    name_a = format_model_name(run_a)
    name_b = format_model_name(run_b)

    if name_a == name_b:
        raise ValueError(f"Cannot compare runs with identical model names: '{name_a}'")

    y_true_a, y_pred_a = run_a.load_targets()
    _, y_pred_b = run_b.load_targets()

    y_true_a = ensure_sample_index_fn(y_true_a, run_a.get(DATA_SCHEMA_ID_COLUMN))
    y_pred_a = ensure_sample_index_fn(y_pred_a, run_a.get(DATA_SCHEMA_ID_COLUMN))
    y_pred_b = ensure_sample_index_fn(y_pred_b, run_b.get(DATA_SCHEMA_ID_COLUMN))

    y_true_a = y_true_a.rename(columns={run_a.get(DATA_SCHEMA_TARGET_COLUMN): TRUE_VALUE_COLUMN_LABEL})
    y_pred_a.columns = [f"{name_a} {PREDICTION_COLUMN_LABEL}"]
    y_pred_b.columns = [f"{name_b} {PREDICTION_COLUMN_LABEL}"]

    df = y_true_a.join(y_pred_a).join(y_pred_b)

    # relabel columns and show which are correct/wrong
    pred_a = f"{name_a} {PREDICTION_COLUMN_LABEL}"
    pred_b = f"{name_b} {PREDICTION_COLUMN_LABEL}"

    p1_correct = df[pred_a] == df[TRUE_VALUE_COLUMN_LABEL]
    p2_correct = df[pred_b] == df[TRUE_VALUE_COLUMN_LABEL]
    same_pred = df[pred_a] == df[pred_b]

    df["Comparison"] = np.select(
        [
            p1_correct & p2_correct,
            p1_correct & ~p2_correct,
            ~p1_correct & p2_correct,
            ~p1_correct & ~p2_correct & same_pred,
            ~p1_correct & ~p2_correct & ~same_pred,

        ],
        [
            "Both correct",
            f"{name_a} correct",
            f"{name_b} correct",
            "Both incorrect - same prediction",
            "Both incorrect - different predictions"
        ],
        default="Unknown"
    )

    col = df.pop("Comparison")
    df.insert(0, "Comparison", col)

    return df.reset_index()

