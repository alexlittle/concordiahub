"""
Base abstract class for all Visualisation adapters
"""

import pandas as pd

from typing import Any, Optional, ClassVar
from abc import ABC, abstractmethod

from xailens import config


class VizAdapter(ABC):
    """
    Base abstract class for all Visualisation adapters
    """

    display_name: str | None = None
    key: ClassVar[str] = None

    compatible_explanations: list[str] = []

    @abstractmethod
    def render(self, explanation: pd.DataFrame, axis_range: Optional[tuple[float, float]]=None) -> list[Any]:
        """
        Abstract method to render the explanation

        Args:
            explanation (pd.DataFrame): explanation dataframe
            axis_range (Optional[tuple[float, float]]): (min, max) bounds for
                the x-axis. If None, Plotly auto-scales the axis.

        Returns:
            list[Any] - list containing the overall explanation if it exists, and a dataframe for tabular display

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    #
    def _top_features(self, explanation: pd.DataFrame, n: int = config.PLOT_MAX_FEATURES) -> pd.DataFrame:
        """
        Helper method to filter to only the largest feature contributions, not all

        Args:
            explanation (pd.DataFrame): explanation dataframe:
            n (int): number of feature contributions to select

        Returns:
            pd.DataFrame - dataframe containing the largest contributions
        """
        return explanation.reindex(explanation["score_normalised_max"].astype(float).abs().nlargest(n).index)
