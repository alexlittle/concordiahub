
from abc import ABC, abstractmethod

class VizAdapter(ABC):

    @abstractmethod
    def render(self, explanation):
        pass
