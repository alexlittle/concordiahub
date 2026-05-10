# xaicompare/runner.py

from .artifacts import ArtifactStore
from .config import RunContext
from .registry.model_registry import get_model_adapter
from .registry.xai_registry import get_xai_adapter

def run(ctx: RunContext):

    # store all the raw data
    artifact_store = ArtifactStore(ctx)
    artifact_store.write_metadata()

    # save model
    artifact_store.write_model()

    # save core data files
    artifact_store.write_data("X_train", ctx.X_train)
    artifact_store.write_data("y_train", ctx.y_train)
    artifact_store.write_data("X_test", ctx.X_test)
    artifact_store.write_data("y_test", ctx.y_test)
    artifact_store.write_data("y_pred", ctx.y_pred)

    # run the XAI methods
    model_adapter = get_model_adapter(ctx.model_type, ctx.model)

    for xai_method in ctx.xai_methods:
        xai_adapter = get_xai_adapter(xai_method)

        results = xai_adapter.explain(model_adapter.prepare(ctx.X_train), ctx.X_train, ctx.X_test)





