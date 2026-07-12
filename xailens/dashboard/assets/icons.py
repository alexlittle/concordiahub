import base64
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=None)
def _load_icon_base64(filename: str) -> str:
    icon_path = Path(__file__).parent / filename
    with open(icon_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def help_icon(tooltip: str) -> str:
    svg_base64 = _load_icon_base64("help-icon.svg")
    return f"""
        <span title="{tooltip}" class="help-icon-wrap">
            <img src="data:image/svg+xml;base64,{svg_base64}" alt="Help" class="help-icon"/>
        </span>
    """

def arrow_icon(alt_text) -> str:
    svg_base64 = _load_icon_base64("arrow-icon.svg")
    return f"""
         <img src="data:image/svg+xml;base64,{svg_base64}" alt="{alt_text}" class="xl-metric-icon" />
        """