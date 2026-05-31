from typing import Any
from abc import ABC, abstractmethod

class VizAdapter(ABC):

    display_name: str | None = None
    compatible_explanations: list[str] = []

    @abstractmethod
    def render_local(self, explanation, sample_id_col) -> Any:
        raise NotImplementedError

    @abstractmethod
    def render_global(self, explanations, sample_id_col) -> Any:
        raise NotImplementedError
