

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix



class ModelAdapter:

    def __init__(self, model, class_names):
        self.model = model

    def predict(self, X):
        raise NotImplementedError



