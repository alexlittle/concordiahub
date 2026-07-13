"""
Defines the run() function - the entry point for a developer to send their model
for processing/storage/view on XAI Lens
"""


import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from dataclasses import fields

from xailens.adapters.explainers.xaimethod_base import XAIMethodAdapter
from xailens.adapters.models.model_base import ModelAdapter
from xailens import strings
from xailens.exceptions import InvalidConfigurationError, MissingDataError
from xailens.artifacts import ArtifactStore
from xailens.entities.context import ModelContext, ModelData
from xailens.registry.model_registry import get_model_adapter
from xailens.registry.xai_registry import get_xai_adapter


def compute_metrics(data: ModelData) -> pd.Series:
    """
    Computes the metrics for the model, accuracy, f1, precision and recall

    Args:
        data (ModelData): dataclass

    Returns:
        pd.Series:

    Raises:
        MissingDataError: if required data is missing
    """
    if data.y_test is not None and data.y_pred is not None:
        instance_id = data.data_schema.instance_id_column
        full_test = data.y_test

        pred_data = data.y_pred.set_index(instance_id).squeeze()
        test_data = full_test.set_index(instance_id).loc[pred_data.index].squeeze()
        series = pd.Series({
            "n_instances": len(test_data),
            "n_failed": len(full_test) - len(test_data),
            "accuracy": round(accuracy_score(test_data, pred_data), 4),
            "f1_macro": round(f1_score(test_data, pred_data, average='macro'), 4),
            "f1_weighted": round(f1_score(test_data, pred_data, average='weighted'),4),
            "precision_macro": round(precision_score(test_data, pred_data, average='macro', zero_division=0), 4),
            "precision_weighted": round(precision_score(test_data, pred_data, average='weighted', zero_division=0), 4),
            "recall_macro": round(recall_score(test_data, pred_data, average='macro', zero_division=0), 4),
            "recall_weighted": round(recall_score(test_data, pred_data, average='weighted', zero_division=0), 4),
        })
        series.index.name = 'metric'
        series.name = 'value'
        return series
    else:
        raise MissingDataError("Cannot compute metrics because y_test and y_pred are required")


def write_global_explanations(xai_method: str,
                              xai_adapter: XAIMethodAdapter,
                              ctx: ModelContext,
                              model_adapter: ModelAdapter,
                              artifact_store: ArtifactStore) -> None:
    """
    Generates, processes and writes global explanations for the model

    Args:
        xai_method (str): XAI method to use
        xai_adapter (XAIMethodAdapter): XAIMethodAdapter to use
        ctx (ModelContext): the ModelContext being used
        model_adapter (ModelAdapter):  the ModelAdapter being used
        artifact_store (ArtifactStore): the ArtifactStore being used

    Returns:
        None
    """
    print(strings.RUN_GLOBAL_EXPLANATIONS.format(xai_method=xai_method))
    if xai_adapter.explanation_file_type == "json":
        results = xai_adapter.explain_global_json(ctx)
        artifact_store.write_explanation(f"exp_global_{xai_method}", results, data_format="df")
    else:
        results = xai_adapter.explain_global_df(ctx, model_adapter.prepare())
        artifact_store.write_explanation(f"exp_global_{xai_method}", results, data_format="df")


def write_local_explanations(xai_method: str,
                              xai_adapter: XAIMethodAdapter,
                              ctx: ModelContext,
                              model_adapter: ModelAdapter,
                              artifact_store: ArtifactStore):
    print(strings.RUN_LOCAL_EXPLANATIONS.format(xai_method=xai_method))
    if xai_adapter.explanation_file_type == "json":
        results = xai_adapter.explain_local_json(ctx)
        artifact_store.write_explanation(f"exp_local_{xai_method}", results, data_format="json")
    else:
        results = xai_adapter.explain_local_df(ctx, model_adapter.prepare())
        artifact_store.write_explanation(f"exp_local_{xai_method}", results, data_format="df")


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

        # Run the global explanations
        write_global_explanations(xai_method, xai_adapter, ctx, model_adapter, artifact_store)

        # Run the local explanations
        write_local_explanations(xai_method, xai_adapter, ctx, model_adapter, artifact_store)

    print(strings.RUN_COMPLETE)