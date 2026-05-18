

from dataclasses import fields

from xai_compare.exceptions import InvalidConfigurationError
from xai_compare.artifacts import ArtifactStore
from xai_compare.config import RunContext
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
        artifact_store.write_data(field.name, getattr(ctx.data, field.name))

    # run the XAI methods
    model_adapter = get_model_adapter(ctx.model_type, ctx.model)

    for xai_method in ctx.xai_methods:
        xai_adapter = get_xai_adapter(xai_method)

        # Run the local explanations
        results = xai_adapter.explain_local(model_adapter.prepare(), ctx.data)
        artifact_store.write_data(f"explanations_{xai_method}", results)




