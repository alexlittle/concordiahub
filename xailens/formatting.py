from datetime import datetime

from xailens.entities.modeleval import ModelEval

def format_datetime(dt_str: str) -> str:
    """For string formatting of the datetime
    Args:
        dt_str (str): datetime string

    Returns:
        String: formatted datetime
    """

    dt = datetime.strptime(dt_str, "%Y%m%d-%H%M%S")
    return dt.strftime("%H:%M %d %b %Y")

def format_model_name(run: ModelEval) -> str:
    """Formats the model name based on run object
    Args:
        run (ModelEval): run object

    Returns:
        String: formatted model name
    """
    if run.get("model_id"):
        parts = []
        parts.append(run.get("model_id"))
        if run.get("model_subtype"):
            parts.append(f"({run.get('model_subtype')})")
        return " ".join(parts)
    else:
        return run.get("model_type")