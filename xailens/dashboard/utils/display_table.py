


def generate_rows_html(df):
    rows = []
    for idx, row in df.iterrows():
        cells = "".join(f"<td>{row[col]}</td>" for col in df.columns)
        rows.append(f"<tr>{cells}</tr>")
    return "\n".join(rows)

def render_dataframe(df):
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