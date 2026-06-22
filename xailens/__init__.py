from xailens.registry.autodiscover import autodiscover_adapters
from xailens.entities.context import ModelContext, ModelData, DataSchema, ResponseSchema
from xailens.runner import run

# Register on package import
autodiscover_adapters()
