import pandas as pd
from abc import ABC, abstractmethod
from typing import Any

from xai_compare.context import ComparisonContext
from xai_compare.services.preprocess import normalise_by_sum


class ComparisonAdapter(ABC):
    """Handles comparison between two XAI explanations."""

    compatible_pairs: set[frozenset[str]]
    visualisations_available = []

    @classmethod
    def get_abs_max(cls, explanation_a: pd.Series, explanation_b: pd.Series, sample_id_column) -> float:
        """ calculate absolute max value from explanation
        required for ensuring charts use the same scales """
        features_a = normalise_by_sum(explanation_a, sample_id_column)
        features_b = normalise_by_sum(explanation_b, sample_id_column)
        a = pd.to_numeric(features_a, errors='coerce')
        b = pd.to_numeric(features_b, errors='coerce')
        diff = a.sub(b, fill_value=0)
        return diff.abs().max()

    @abstractmethod
    def render_comparison(self, ctx: ComparisonContext, viz_type) -> Any:
        raise NotImplementedError