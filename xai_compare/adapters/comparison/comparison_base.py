import pandas as pd
from abc import ABC, abstractmethod
from typing import Any

from xai_compare.run import Run
from xai_compare.services.preprocess import normalise_values


class ComparisonAdapter(ABC):
    """Handles comparison between two XAI explanations."""

    compatible_pairs: set[frozenset[str]]
    visualisations_available = []

    @classmethod
    def get_abs_max(cls, explanation_a: pd.Series, explanation_b: pd.Series, sample_id_column) -> float:
        """ calculate absolute max value from explanation
        required for ensuring charts use the same scales """
        features_a = normalise_values(explanation_a, sample_id_column)
        features_b = normalise_values(explanation_b, sample_id_column)
        a = pd.to_numeric(features_a, errors='coerce')
        b = pd.to_numeric(features_b, errors='coerce')
        diff = a.sub(b, fill_value=0)
        return diff.abs().max()

    @abstractmethod
    def render_comparison(self,
                          explanation_a: Any,
                          explanation_b: Any,
                          run_a: Run,
                          run_b: Run,
                          axis_range=None) -> Any:
        raise NotImplementedError