
from dataclasses import fields

from xai_compare.exceptions import InvalidConfigurationError
from xai_compare.encoder import XAITextEncoder
from xai_compare.artifacts import ArtifactStore
from xai_compare.context import RunContext
from xai_compare.registry.model_registry import get_model_adapter
from xai_compare.registry.xai_registry import get_xai_adapter

def run(ctx: RunContext):
    # check ctx provided
    if ctx is None:
        raise InvalidConfigurationError("Context (ctx) is required")

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
        results = xai_adapter.explain_local(ctx.data, model_adapter.prepare())
        artifact_store.write_explanation(f"explanations_{xai_method}", results)

    '''
    run_exp = RunExplorer(artifact_store)
    print(run_exp.get_confusion_matrix())
    print(run_exp.get_accuracy_score())
    print(run_exp.get_classification_report())
    '''
