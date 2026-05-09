# xaicompare/adapters/explainers/explainer_base.py
from abc import ABC, abstractmethod



class ExplainerAdapter(ABC):

    def __init__(self, model_adapter, config):
        self.m = model_adapter
        self.config = config or {}

    @abstractmethod
    def limitation_text(self):
        pass

    @abstractmethod
    def local_explanations(self, x_row):
        pass



