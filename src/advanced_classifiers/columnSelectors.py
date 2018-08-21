from sklearn.base import BaseEstimator, TransformerMixin
from src.features.fasttext_features import *
import numpy as np

class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

class DummySelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

def get_fast_text_model(x, y):
    input_dim = np.max(x) + 1
    model = create_model(input_dim)
    hist = model.fit(x, y,
                     batch_size=16,
                     validation_data=None,
                     epochs=16,
                     callbacks=[EarlyStopping(patience=4, monitor='val_loss')])
    return hist.model


class FastTextPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, docstrain, ytrain):
        return self

    def transform(self, X):
        return pd.DataFrame(self.model.predict_proba(X), columns=['a1', 'a2', 'a3'])


class FastTextPreTrain(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return create_docs(X[self.key], train_mode=False)



class FastTextEstimator(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key, xtrain, ytrain, xvalid, yvalid):
        self.key = key
        self.xtrain_docs = create_docs(xtrain[self.key], train_mode=False)
        self.ytrain = ytrain
        self.xvalid_docs = create_docs(xvalid[self.key], train_mode=False)
        self.yvalid = yvalid

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        all_docs = np.append(self.xtrain_docs, self.xvalid_docs, axis=0)
        max = np.max(all_docs) + 1
        features = get_fasttext_features1(self.xtrain_docs, self.ytrain, self.xvalid_docs, self.yvalid, max)[0]
        return pd.DataFrame(features, columns=['a1', 'a2', 'a3'])
