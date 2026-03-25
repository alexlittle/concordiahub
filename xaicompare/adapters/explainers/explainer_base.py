# xaicompare/adapters/explainer_base.py
from abc import ABC, abstractmethod

import numpy as np


class ExplainerAdapter(ABC):

    def __init__(self, model_adapter, config):

        self.m = model_adapter
        self.config = config or {}

    @abstractmethod
    def limitation_text(self):
        raise NotImplementedError

    @abstractmethod
    def global_importance(self, x, rows_limit):
        raise NotImplementedError

    @abstractmethod
    def local_explanations(self, x_row):
        raise NotImplementedError



