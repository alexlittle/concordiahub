import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from dataclasses import fields

from xailens import strings
from xailens.exceptions import InvalidConfigurationError, MissingDataError
from xailens.encoder import XAITextEncoder
from xailens.artifacts import ArtifactStore
from xailens.entities.context import ModelContext
from xailens.registry.model_registry import get_model_adapter
from xailens.registry.xai_registry import get_xai_adapter


def compute_metrics(data):
    if (data.y_test is not None or data.y is not None) and data.y_pred is not None:
        instance_id = data.data_schema.instance_id_column
        full_test = data.y_test if data.y_test is not None else data.y

        pred_data = data.y_pred.set_index(instance_id).squeeze()
        test_data = full_test.set_index(instance_id).loc[pred_data.index].squeeze()
        series = pd.Series({
            "n_instances": len(test_data),
            "n_failed": len(full_test) - len(test_data),
            "accuracy": round(accuracy_score(test_data, pred_data), 4),
            "f1_macro": round(f1_score(test_data, pred_data, average='macro'), 4),
            "f1_weighted": round(f1_score(test_data, pred_data, average='weighted'),4)
        })
        series.index.name = 'metric'
        series.name = 'value'
        return series
    else:
        raise MissingDataError("Cannot compute metrics because y or y_test, and y_pred are required")

def run(ctx: ModelContext):
    print(strings.RUN_STARTING)

    # check ctx provided
    if ctx is None:
        raise InvalidConfigurationError("Context (ctx) is required")

    artifact_store = ArtifactStore(ctx)

    # Compute and store metrics
    print(strings.RUN_COMPUTING_METRICS)
    metrics = compute_metrics(ctx.data)
    artifact_store.write_metrics(metrics)

    # store all the metadata
    print(strings.RUN_SAVING_METADATA)
    artifact_store.write_metadata()

    # save model
    print(strings.RUN_SAVING_MODEL_DATA)
    artifact_store.write_model()

    # save RunData as files
    for field in fields(ctx.data):
        if field.name != "data_schema":
            artifact_store.write_data(field.name, getattr(ctx.data, field.name))

    # run the XAI methods
    print(strings.RUN_RUNNING_XAI_METHODS)
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

        # Run the global explanations
        print(strings.RUN_GLOBAL_EXPLANATIONS.format(xai_method=xai_method))

        # Run the local explanations
        print(strings.RUN_LOCAL_EXPLANATIONS.format(xai_method=xai_method))
        if xai_adapter.explanation_file_type == "json":
            results = xai_adapter.explain_local_json(ctx)
            artifact_store.write_explanation(f"exp_local_{xai_method}", results, type="json")
        else:
            results = xai_adapter.explain_local_df(ctx, model_adapter.prepare())
            artifact_store.write_explanation(f"exp_local_{xai_method}", results, type="df")

    print(strings.RUN_COMPLETE)