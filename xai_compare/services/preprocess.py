import pandas as pd


def normalise_by_max(explanation: pd.Series, sample_id_column) -> pd.Series:
    """Normalise to [-1, 1] by dividing by absolute max.
    Use for single explanation display."""
    features = explanation.drop(index=sample_id_column, errors='ignore')
    numeric = pd.to_numeric(features, errors='coerce')
    abs_max = numeric.abs().max()
    return numeric / abs_max if abs_max != 0 else numeric

def normalise_by_sum(explanation: pd.Series, sample_id_column) -> pd.Series:
    """Normalise by sum of absolute values and sort.
    Use for comparison display — makes cross-method contributions comparable."""
    features = explanation.drop([sample_id_column, "class"], errors="ignore")
    normalised = features / features.abs().sum()
    return normalised.sort_values(ascending=True)

