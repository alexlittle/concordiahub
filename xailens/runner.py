
from sklearn.metrics import accuracy_score, f1_score
from dataclasses import fields

from xailens.exceptions import InvalidConfigurationError
from xailens.encoder import XAITextEncoder
from xailens.artifacts import ArtifactStore
from xailens.context import RunContext, RunMetrics
from xailens.registry.model_registry import get_model_adapter
from xailens.registry.xai_registry import get_xai_adapter


def compute_metrics(data):
    if (data.y_test is not None or data.y is not None) and data.y_pred is not None:
        sample_id = data.data_schema.sample_id_column
        full_test = data.y_test if data.y_test is not None else data.y

        pred_data = data.y_pred.set_index(sample_id)
        test_data = full_test.set_index(sample_id).loc[pred_data.index]

        return RunMetrics(
            n_samples=len(test_data),
            n_failed=len(full_test) - len(test_data),
            accuracy=round(accuracy_score(test_data, pred_data), 4),
            f1_macro=round(f1_score(test_data, pred_data, average='macro'), 4),
            f1_weighted=round(f1_score(test_data, pred_data, average='weighted'), 4))
    else:
        return RunMetrics()

def run(ctx: RunContext):
    # check ctx provided
    if ctx is None:
        raise InvalidConfigurationError("Context (ctx) is required")

    ctx.metrics = RunMetrics(ctx.data)

    # store all the raw data
    artifact_store = ArtifactStore(ctx)
    artifact_store.write_metadata()

    # save model
    artifact_store.write_model()

    # save RunData as files
    for field in fields(ctx.data):
        if field.name != "data_schema":
            artifact_store.write_data(field.name, getattr(ctx.data, field.name))

    # run the XAI methods
    model_class = get_model_adapter(ctx.model_type)
    model_adapter = model_class(ctx.model)

    for xai_method in ctx.xai_methods:
        xai_class = get_xai_adapter(xai_method)
        xai_adapter = xai_class()

        # process feature names if necessary
        if xai_adapter.requires_encoded_features:
            encoder = XAITextEncoder()
            if ctx.data.x is not None:
                df = ctx.data.x
            elif ctx.data.x_train is not None:
                df = ctx.data.x_train
            encoded_features = encoder.encode_feature_names(df)
            artifact_store.write_data("feature_name_embeddings", encoded_features)

        # Run the local explanations
        if xai_adapter.explanation_file_type == "json":
            results = xai_adapter.explain_local_json(ctx)
            artifact_store.write_explanation(f"explanations_{xai_method}", results, type="json")
        else:
            results = xai_adapter.explain_local_df(ctx, model_adapter.prepare())
            artifact_store.write_explanation(f"explanations_{xai_method}", results, type="df")

