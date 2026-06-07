from xai_compare.registry.autodiscover import autodiscover_adapters
from xai_compare.context import RunContext, RunData, DataSchema, ResponseSchema
from xai_compare.runner import run

# Register on package import
autodiscover_adapters()
