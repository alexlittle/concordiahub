import pandas as pd
from abc import ABC, abstractmethod

from xai_compare.config import RunData

class XAIMethodAdapter(ABC):
    """
    Abstract class for XAI method adapter.
    """

    @abstractmethod
    def explain(self, model, data: RunData) -> pd.DataFrame:
        pass



