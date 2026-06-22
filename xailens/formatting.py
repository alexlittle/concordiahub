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