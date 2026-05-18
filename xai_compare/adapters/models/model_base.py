from abc import ABC, abstractmethod


class ModelAdapter(ABC):

    def __init__(self, model=None):
        self.model = model

    @abstractmethod
    def prepare(self):
        pass



