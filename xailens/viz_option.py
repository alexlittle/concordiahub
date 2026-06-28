from dataclasses import dataclass

@dataclass
class VizOption:
    display_name: str
    method: str | None = None