# xaicompare/__init__.py
from xaicompare.registry.autodiscover import autodiscover_adapters

# Register on package import
autodiscover_adapters()
