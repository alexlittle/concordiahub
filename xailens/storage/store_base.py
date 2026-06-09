'''
Abstract class for storing data and model
'''

import pandas as pd
from abc import ABC, abstractmethod


class DataStore(ABC):
    @abstractmethod
    def save_json(self, filename, data): pass

    @abstractmethod
    def save_dataframe(self, filename, data): pass

    @abstractmethod
    def save_model(self, filename, model): pass

    @abstractmethod
    def save_explanation(self, filename, data, type): pass

    @abstractmethod
    def load_dataframe(self, filename) -> pd.DataFrame: pass

    @abstractmethod
    def load_json(self, filename) -> pd.DataFrame: pass