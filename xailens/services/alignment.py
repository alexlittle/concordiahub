import pandas as pd

from xailens.dashboard.services import get_explorer
from xailens.services.comparison import build_comparison_df
from xailens.registry.xai_registry import get_xai_adapter
from xailens import config, schema

def compute_feature_differences(explanation_a, explanation_b):
    a = explanation_a.set_index("feature_name")["score_normalised_max"]
    b = explanation_b.set_index("feature_name")["score_normalised_max"]
    features = a.index.union(b.index)

    matched = []
    only_in_a = []
    only_in_b = []
    for feature in features:
        a_present = feature in a.index
        b_present = feature in b.index
        if a_present and b_present:
            diff = abs(a[feature] - b[feature])
            matched.append((feature, diff))
        elif a_present:
            only_in_a.append((feature, f"{a[feature]:.4f}"))
        else:
            only_in_b.append((feature, f"{b[feature]:.4f}"))
    return matched, only_in_a, only_in_b


def score_to_level(score: float) -> int:
    for i, threshold in enumerate(config.ALIGNMENT_THRESHOLDS):
        if score < threshold:
            return len(schema.ALIGNMENT_LEVELS) - i
    return 1


def compute_overall_alignment(explanation_a, explanation_b) -> dict:
    matched, only_in_a, only_in_b = compute_feature_differences(explanation_a, explanation_b)

    mean_diff = sum(diff for _, diff in matched) / len(matched) if matched else None
    level = score_to_level(mean_diff) if mean_diff is not None else None

    return {
        "alignment_score": mean_diff,
        "alignment_level": level,
        "n_matched": len(matched),
        "n_only_a": len(only_in_a),
        "n_only_b": len(only_in_b),
    }


def main():

    explorer = get_explorer()
    modeleval_a = explorer.get_modeleval_by_uuid("4dd56b8c-d22e-4d1e-9cc3-c9dbb696c105")
    modeleval_b = explorer.get_modeleval_by_uuid("56eae897-e4fd-43bb-927e-c4c7fd67d526")

    method_a = modeleval_a.metadata.get("xai_methods")[0]
    method_b = modeleval_b.metadata.get("xai_methods")[0]

    adapter_a = get_xai_adapter(method_a)
    adapter_b = get_xai_adapter(method_b)

    df = build_comparison_df(modeleval_a, modeleval_b)
    print(df.columns.tolist())
    print(len(df))

    rows = []
    for instance_id in df["instance_uuid"]:
        explanation_a = adapter_a.load_explanation(modeleval_a, instance_id)
        explanation_b = adapter_b.load_explanation(modeleval_b, instance_id)
        result = compute_overall_alignment(explanation_a, explanation_b)
        result["instance_uuid"] = instance_id
        rows.append(result)

    alignment_df = pd.DataFrame(rows)
    print(alignment_df)

if __name__ == "__main__":
    main()