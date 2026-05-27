
from xai_compare.dashboard.components.formatting import format_model_name

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

    y_true_a = ensure_sample_index_fn(y_true_a, run_a.get('data_schema.sample_id_column'))
    y_pred_a = ensure_sample_index_fn(y_pred_a, run_a.get('data_schema.sample_id_column'))
    y_pred_b = ensure_sample_index_fn(y_pred_b, run_b.get('data_schema.sample_id_column'))

    y_true_a = y_true_a.rename(columns={run_a.get('data_schema.target_column'): "True Value"})
    y_pred_a.columns = [f"{name_a} Prediction"]
    y_pred_b.columns = [f"{name_b} Prediction"]

    df = y_true_a.join(y_pred_a).join(y_pred_b)

    return df.reset_index()

