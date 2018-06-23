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
#print('Accuracy on the training set: {:.3f}'.format(log_reg.score(xtrain, ytrain)))
#print('Accuracy on the test set: {:.3f}'.format(log_reg.score(xtest, ytest)))
predictions = log_reg.predict_proba(xvalid_tfv)
preds = log_reg.predict(xvalid_tfv)

print("logloss: %0.3f " % multiclass_logloss(ytest, predictions))
print("business friendly output: %0.3f" % (np.sum(preds == ytest) / len(ytest)))