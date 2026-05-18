import pandas as pd
from abc import ABC, abstractmethod


class XAIMethodAdapter(ABC):
    """
    Abstract class for XAI method adapter.
    """

    @abstractmethod
    def explain(self, model, X_train, X_test) -> pd.DataFrame:
        pass



