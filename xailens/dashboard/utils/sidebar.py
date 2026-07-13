"""
Functions for rendering the streamlit sidebar
"""

import base64

from pathlib import Path

from xailens._version import __version__
from xailens import strings

def render_header() -> str:
    """
    Generates HTML code for the sidebar header

    Returns:
        str: HTML code
    """
    logo_path = Path(__file__).parent.parent / "assets" / "xai-lens-logo.svg"

    with open(logo_path, "rb") as f:
        svg_base64 = base64.b64encode(f.read()).decode()

    return f"""
        <a href="/" style="text-decoration: none;" target="_self">
            <div class="logo-container">
                <img src="data:image/svg+xml;base64,{svg_base64}" alt="XAI Lens Logo"/>
            </div>
            <h1 class="logo">{strings.APP_NAME}</h1>
        </a>
    """

def render_footer() -> str:
    """
    Generates HTML code for the sidebar footer

    Returns:
        str: HTML code
    """

    return f"""
        <div class="sidebar-footer">
            XAI Lens v{__version__}<br/>
            <a href="https://github.com/alexlittle/xailens">Source Code (GitHub)</a> 
            &middot; 
            Documentation
        </div>
    """

