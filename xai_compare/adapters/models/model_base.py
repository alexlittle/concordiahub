from abc import ABC, abstractmethod


class ModelAdapter(ABC):

    display_name: str | None = None

    def __init__(self, model=None):
        self.model = model

    def prepare(self):
        return self.model



