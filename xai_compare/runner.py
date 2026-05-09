# xaicompare/runner.py

from .artifacts import ArtifactStore
from .config import RunContext


def run(ctx: RunContext):

    artifact_store = ArtifactStore(ctx)
    artifact_store.write_metadata()




