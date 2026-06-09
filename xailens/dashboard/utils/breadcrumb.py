
from xailens.run import Run
from xailens.dashboard.utils.formatting import format_model_name


def display_breadcrumbs(run_a: Run=None, run_b: Run=None, sample_id:str = None) -> str:
    parts = []
    parts.append("<a href='/' target='_self'>Home</a>")

    if run_a and run_b:
        if sample_id:
            parts.append(f"""
            <a href='/overview?runs={run_a.get('run_uuid')},{run_b.get('run_uuid')}' target='_self'>
            {format_model_name(run_a)} &amp; {format_model_name(run_b)}</a>
            """)
        else:
            parts.append(f"""
                        {format_model_name(run_a)} &amp; {format_model_name(run_b)}
                        """)

    if sample_id:
        parts.append(f"{sample_id}")

    return " &nbsp;&gt;&nbsp; ".join(parts)