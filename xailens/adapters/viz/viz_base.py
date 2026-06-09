import pandas as pd

from typing import Any
from abc import ABC, abstractmethod


class VizAdapter(ABC):

    display_name: str | None = None
    compatible_explanations: list[str] = []

    @abstractmethod
    def render_local(self, explanation: pd.DataFrame, axis_range=None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def render_global(self, explanations) -> Any:
        raise NotImplementedError
