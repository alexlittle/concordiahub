import pandas as pd
from markdown import markdown

def generate_rows_html(df: pd.DataFrame) -> str:
    rows = []
    for idx, row in df.iterrows():
        cells = "".join(f"<td>{row[col]}</td>" for col in df.columns)
        rows.append(f"<tr>{cells}</tr>")
    return "\n".join(rows)

def render_dataframe(df: pd.DataFrame) -> str:
        rows_html = generate_rows_html(df)
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


def render_series(series: pd.Series) -> str:
    rows = [
        f"<tr><td>{name}</td><td>{markdown(str(value))}</td></tr>" for name, value in series.items()
    ]
    rows_html = "\n".join(rows)
    return f"""
        <table class="report-table">
            <thead>
                <tr>
                    <th nowrap>Feature Name</th>
                    <th nowrap>Value</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """