

from dataclasses import fields

from .artifacts import ArtifactStore
from .config import RunContext
from .registry.model_registry import get_model_adapter
from .registry.xai_registry import get_xai_adapter

def run(ctx: RunContext):
    # check ctx provided
    if ctx is None:
        raise ValueError("Context (ctx) is required")

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

        results = xai_adapter.explain(model_adapter.prepare(ctx.data.x_train), ctx.data.x_train, ctx.data.x_test)
        print(results)




