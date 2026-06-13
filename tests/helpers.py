import pandas as pd

def make_explanation(scores: dict) -> pd.DataFrame:
    values = list(scores.values())
    abs_sum = sum(abs(v) for v in values)
    abs_max = max(abs(v) for v in values)
    return pd.DataFrame([
        {
            "feature_name": k,
            "score": v,
            "score_normalised_sum": v / abs_sum if abs_sum != 0 else v,
            "score_normalised_max": v / abs_max if abs_max != 0 else v,
            "reasoning": None
        }
        for k, v in scores.items()
    ])