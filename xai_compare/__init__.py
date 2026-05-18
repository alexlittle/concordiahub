# xaicompare/__init__.py
from xai_compare.registry.autodiscover import autodiscover_adapters
from xai_compare.config import RunContext, RunData
from xai_compare.runner import run

# Register on package import
autodiscover_adapters()
