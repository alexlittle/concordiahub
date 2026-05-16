# ./storage/store_base.py
'''
Abstract class for storing data and model
'''

from abc import ABC, abstractmethod


class DataStore(ABC):
    @abstractmethod
    def save_json(self, filename, data): pass

    @abstractmethod
    def save_dataframe(self, filename, data): pass

    @abstractmethod
    def save_ndarray(self, filename, data): pass

    @abstractmethod
    def save_pdseries(self, filename, data): pass

    @abstractmethod
    def save_model(self, filename, model): pass