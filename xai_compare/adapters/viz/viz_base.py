from typing import Any
from abc import ABC, abstractmethod

class VizAdapter(ABC):

    @abstractmethod
    def render_local(self, explanation) -> Any:
        raise NotImplementedError

    @abstractmethod
    def render_global(self, explanations) -> Any:
        raise NotImplementedError
