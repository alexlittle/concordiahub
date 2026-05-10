# xaicompare/runner.py

from .artifacts import ArtifactStore
from .config import RunContext


def run(ctx: RunContext):

    # store all the raw data
    artifact_store = ArtifactStore(ctx)
    artifact_store.write_metadata()

    # model
    artifact_store.write_model()

    # core data files etc
    artifact_store.write_data("X_train", ctx.X_train)
    artifact_store.write_data("y_train", ctx.y_train)
    artifact_store.write_data("X_test", ctx.X_test)
    artifact_store.write_data("y_test", ctx.y_test)
    artifact_store.write_data("y_pred", ctx.y_pred)

    # run the XAI methods






