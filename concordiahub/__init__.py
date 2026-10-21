from concordiahub.registry.autodiscover import autodiscover_adapters
from concordiahub.entities.context import ModelContext, ModelData, DataSchema, ResponseSchema
from concordiahub.runner import run

# Register on package import
autodiscover_adapters()
