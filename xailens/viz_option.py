"""
Dataclass for VizOption
"""

from dataclasses import dataclass

@dataclass
class VizOption:
    """
    Dataclass for VizOption - with a display name and method
    """
    display_name: str
    method: str | None = None