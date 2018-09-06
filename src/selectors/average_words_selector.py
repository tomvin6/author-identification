import pandas as pd
import os
from sklearn.base import BaseEstimator, TransformerMixin
from src.features.writing_style_features import preprocess_text


class AverageWordsSelector(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.max = 0
        self.min = 0
        self.mean = 0

    def fit(self, df, y=None):
        self.count = df['text'].apply(word_count, convert_dtype=True)
        self.max = self.count.max()
        self.min = self.count.min()
        self.mean = self.count.mean()
        return self

    def transform(self, df):
        return pd.DataFrame(self.count.apply(lambda x: (self.mean - x) / (self.max - self.min)))



class MetaStyleSelector(BaseEstimator, TransformerMixin):

    def __init__(self, name):
        self.name = name

    def fit(self, df):
        return self

    def transform(self, df):
        if not os.path.isfile(self.name):
            df = preprocess_text(df)
            df.to_pickle(self.name)
        return pd.read_pickle(self.name)
