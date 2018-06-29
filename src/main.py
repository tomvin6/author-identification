import os

import pandas as pd
import numpy as np
#import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from nltk.corpus import stopwords
from src.evaluations.logloss import *
stop_words = stopwords.words('english')


# load data
train = pd.read_csv('..' + os.sep + 'input' + os.sep + 'train.csv')
test = pd.read_csv('..' + os.sep + 'input' + os.sep + 'test.csv')
# sample = pd.read_csv('..' + os.sep + 'input' + os.sep + 'sample_submission.csv')


# We use the LabelEncoder from scikit-learn to convert text labels to integers, 0,1 2
# scikit-learn only handle real numbers (there is a default conversion if not converted)
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train.author.values)  # y is the new train classifications vector

# we split the train data into: train, test sets.
# 10% of the train data will be used as test set.
xtrain, xtest, ytrain, ytest = train_test_split(train.text.values, y,
                                                stratify=y,
                                                random_state=42,
                                                test_size=0.1, shuffle=True)

print(xtrain.shape)
print(xtest.shape)

print("TF-IDF + LOG REG")
# TF-IDF feature calculations
tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(xtrain) + list(xtest))  # Learn vocabulary and idf from training set.
# list of -> (# of sentence, occurred words number    tf-idf-score)
xtrain_tfv = tfv.transform(xtrain)  # create sparse matrix with tf-idf probs
xvalid_tfv = tfv.transform(xtest)

# Fitting a simple Logistic Regression on TFIDF
log_reg = LogisticRegression(C=1.0)
log_reg.fit(xtrain_tfv, ytrain)  # execute train for Log regression model


predictions = log_reg.predict_proba(xvalid_tfv)
preds = log_reg.predict(xvalid_tfv)
print("logloss: %0.3f " % multiclass_logloss(ytest, predictions))
print("business friendly output: %0.3f" % (np.sum(preds == ytest) / len(ytest)))

# Fitting a simple Naive Bayes on TFIDF
print("NAIVE BAYES")
clf = MultinomialNB()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)
print("logloss: %0.3f " % multiclass_logloss(ytest, predictions))
print("business friendly output: %0.3f" % (np.sum(preds == ytest) / len(ytest)))


# WORD COUNT feature
print("WORD-COUNT + LOG REG")
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')
# Fitting Count Vectorizer to both training and test sets (semi-supervised learning)
ctv.fit(list(xtrain) + list(xtest))
xtrain_ctv =  ctv.transform(xtrain)
xvalid_ctv = ctv.transform(xtest)

# Fitting a simple Logistic Regression on Counts
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_ctv, ytrain)
preds = clf.predict(xvalid_ctv)
predictions = clf.predict_proba(xvalid_ctv)

print ("logloss: %0.3f " % multiclass_logloss(ytest, predictions))
print("business friendly output: %0.3f" % (np.sum(preds == ytest) / len(ytest)))


# Apply SVD, I chose 120 components. 120-200 components are good enough for SVM model.
print("SVM + TFIDF")
svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(xtrain_tfv)
xtrain_svd = svd.transform(xtrain_tfv)
xvalid_svd = svd.transform(xvalid_tfv)

# Scale the data obtained from SVD. Renaming variable to reuse without scaling.
scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xvalid_svd_scl = scl.transform(xvalid_svd)

# Fitting a simple SVM
clf = SVC(C=1.0, probability=True) # since we need probabilities
clf.fit(xtrain_svd_scl, ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(ytest, predictions))
print("business friendly output: %0.3f" % (np.sum(preds == ytest) / len(ytest)))