import pandas as pd

from typing import Any
from abc import ABC, abstractmethod

from xailens import config

class VizAdapter(ABC):

    display_name: str | None = None
    compatible_explanations: list[str] = []

    @abstractmethod
    def render_local(self, explanation: pd.DataFrame,  axis_range=None) -> list[Any]:
        raise NotImplementedError

    # filter to show only the largest contributors, not all
    def _top_features(self, explanation: pd.DataFrame, n: int = config.PLOT_MAX_FEATURES) -> pd.DataFrame:
        return explanation.reindex(explanation["score_normalised_max"].astype(float).abs().nlargest(n).index)
