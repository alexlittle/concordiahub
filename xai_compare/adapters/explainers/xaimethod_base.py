from abc import ABC, abstractmethod


class XAIMethodAdapter(ABC):

    @abstractmethod
    def explain(self, model, X_train, X_test):
        pass



