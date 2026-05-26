


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



def build_comparison_df(run_a, run_b):
    y_true_a, y_pred_a = run_a.load_targets()
    _, y_pred_b = run_b.load_targets()

    y_true_a = ensure_sample_index(y_true_a)
    y_pred_a = ensure_sample_index(y_pred_a)
    y_pred_b = ensure_sample_index(y_pred_b)

    y_true_a = y_true_a.rename(columns={"target": "y_true"})
    y_pred_a.columns = ["pred_A"]
    y_pred_b.columns = ["pred_B"]

    df = y_true_a.join(y_pred_a).join(y_pred_b)

    return df.reset_index()

