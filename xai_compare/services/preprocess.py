


def normalise_values(explanation, sample_id_column):
    features = explanation.drop([sample_id_column, "class"], errors="ignore")

    # Normalise by sum of absolute values
    features = features / features.abs().sum()

    # Sort all features by value
    return features.sort_values(ascending=True)

