
from datetime import datetime

from xai_compare.run import Run

def format_datetime(dt_str: str) -> str:
    """For string formatting of the datetime
    Args:
        dt_str (str): datetime string

    Returns:
        String: formatted datetime
    """

    dt = datetime.strptime(dt_str, "%Y%m%d-%H%M%S")
    return dt.strftime("%H:%M %d %b %Y")

def format_model_name(run: Run) -> str:
    """Formats the model name based on run object
    Args:
        run (Run): run object

    Returns:
        String: formatted model name
    """
    parts = []

    if run.get("model_id" ):
        parts.append(run.get("model_id"))

    if run.get("model_type"):
        parts.append(run.get("model_type"))

    if run.get("model_subtype"):
        parts.append(f"({run.get('model_subtype')})")

    return " - ".join(parts)


