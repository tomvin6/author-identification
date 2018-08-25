import gensim
import os
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.utils import simple_preprocess
from sklearn.base import BaseEstimator, TransformerMixin


def tag_docs(docs):
    tagged = docs.apply(lambda r: TaggedDocument(words=simple_preprocess(r['text']), tags=[r.id]), axis=1)
    return tagged


def train_doc2vec_model(tagged_docs, size, epochs, window=1):
    sents = tagged_docs.values
    doc2vec_model = Doc2Vec(sents, vector_size=size, window=window, epochs=epochs, dm=1)
    return doc2vec_model


def vec_for_learning(doc2vec_model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps=20)) for doc in sents])
    return regressors


class Doc2VecSelector(BaseEstimator, TransformerMixin):

    def __init__(self, model_data_name, feature_size, epochs_number):
        self.model_data_name = model_data_name
        self.feature_size = feature_size
        self.epochs_number = epochs_number

    def fit(self, docs, y=None):
        self.train_tagged = tag_docs(docs)
        # Train or Load model if exist
        if not os.path.isfile(self.model_data_name):
            model = train_doc2vec_model(self.train_tagged , self.feature_size, self.epochs_number)
            # saving the created model
            model.save(self.model_data_name)
            print('model saved')

        self.d2v_model = gensim.models.doc2vec.Doc2Vec.load(self.model_data_name)
        return self

    def transform(self, X):
        return vec_for_learning(self.d2v_model, self.train_tagged)