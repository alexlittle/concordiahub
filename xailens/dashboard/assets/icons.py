"""
Helper functions for SVG icons
"""

import base64
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=None)
def _load_icon_base64(filename: str) -> str:
    """
    Load an icon from a base64 encoded file

    Args:
        filename (str): filename of the icon

    Returns:
        str: base64 encoded icon
    """
    icon_path = Path(__file__).parent / filename
    with open(icon_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def help_icon(tooltip: str) -> str:
    """
    Help icon

    Args:
        tooltip (str): tooltip to display in hover icon

    Returns:
        str: HTML to display help icon
    """
    svg_base64 = _load_icon_base64("help-icon.svg")
    return f"""
        <span title="{tooltip}" class="help-icon-wrap">
            <img src="data:image/svg+xml;base64,{svg_base64}" alt="Help" class="help-icon"/>
        </span>
    """

def arrow_icon(alt_text: str) -> str:
    """
    Arrow icon

    Args:
        alt_text (str): alt text to display

    Returns:
        str: HTML to display arrow icon
    """
    svg_base64 = _load_icon_base64("arrow-icon.svg")
    return f"""
         <img src="data:image/svg+xml;base64,{svg_base64}" alt="{alt_text}" class="xl-metric-icon" />
        """