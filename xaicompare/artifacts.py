# xaicompare/artifacts.py
from pathlib import Path

class ArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def write_parquet(self, name, df):
        out = self.root / name
        df.to_parquet(out, index=False)