import pandas as pd

from xailens import strings
from xailens.dashboard import settings

def _display_value(val) -> str:
    if val in (0, 1, True, False) and not isinstance(val, float):
        return str(bool(val))
    return str(val)


def generate_rows_html(df: pd.DataFrame,
                       query_params:dict,
                       true_label_col:str = "true_label",
                       id_col:str = None,
                       skip_cols:str = None):
    skip_cols = skip_cols or []
    id_display_col = id_col or strings.DISPLAY_NAMES["instance_uuid"]
    pred_cols = [c for c in df.columns if c not in [true_label_col, id_display_col] + skip_cols]
    rows = []
    for idx, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if col == id_display_col:
                query_string = "&".join(f"{k}={v}" for k, v in query_params.items())
                cells.append(f'<td><a href="{settings.INSTANCE_DETAIL_FILENAME}?{query_string}&instance_uuid={val}">{val}</a></td>')
            elif col in pred_cols:
                css_class = "pred-correct" if val == row[true_label_col] else "pred-incorrect"
                cells.append(f'<td class="{css_class}">{_display_value(val)}</td>')
            else:
                cells.append(f"<td>{_display_value(val)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return "\n".join(rows)

def render_instances_df(df:pd.DataFrame,
                        query_params:dict = None,
                        true_label_col:str = "true_label",
                        id_col:str = None,
                        skip_cols:str = None):
        rows_html = generate_rows_html(df, query_params, true_label_col, id_col, skip_cols)
        header = "".join(f"<th nowrap>{col}</th>" for col in df.columns)
        return f"""
        <table class="report-table">
            <thead>
                <tr>
                    {header}
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """

