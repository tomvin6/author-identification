import gensim
import os
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from src.features.writing_style_features import *


class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, df, index):
        self.index = index
        self.df = df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(self.df[:, self.index])