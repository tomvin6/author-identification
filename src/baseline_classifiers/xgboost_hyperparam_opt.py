import os
import pandas as pd
from src.utils.input_reader import *
from src.baseline_classifiers.tf_idf import *
from src.baseline_classifiers.svm_tfidf import *
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import xgboost as xgb
# import xgboost as xgb
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

def run_xgboost_random_search():
    stop_words = stopwords.words('english')
    print("baseline_classifiers classifier")
    print("Algorithm: XGboost with hyperparam optimization (grid search) ")
    print("Features: TF-IDF")

    # LOAD DATA
    path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
    xtrain, xvalid, ytrain, yvalid = train_vali_split(train_df)

    xtrain_tfv, xvalid_tfv = get_dfidf_features(xtrain, xvalid)

    # grid-search model xgboost
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }

    folds = 3
    param_comb = 5  # increase!!
    mll_scorer = metrics.make_scorer(multiclass_logloss, greater_is_better=False, needs_proba=True)

    clf = xgb.XGBClassifier(n_estimators=200, nthread=1, learning_rate=0.02)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    random_search = RandomizedSearchCV(clf, param_distributions=params, n_iter=param_comb, scoring=mll_scorer, n_jobs=4,
                                       cv=skf.split(xtrain_tfv, ytrain), verbose=3, random_state=1001)

    random_search.fit(xtrain_svd, ytrain)  # xtrain_tfv

    predictions = random_search.predict_proba(xvalid_tfv)
    predictions_classes = random_search.predict(xvalid_tfv)

    print("random search xgboost (tf-ifd,svd) logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
    print(
        "random search xgboost business friendly output: %0.3f" % (np.sum(predictions_classes == yvalid) / len(yvalid)))


if __name__ == '__main__':
    stop_words = stopwords.words('english')
    print("baseline_classifiers classifier")
    print("Algorithm: XGboost with hyperparam optimization (grid search) ")
    print("Features: TF-IDF")

    # LOAD DATA
    path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
    xtrain, xvalid, ytrain, yvalid = train_vali_split(train_df)

    xtrain_tfv, xvalid_tfv = get_dfidf_features(xtrain, xvalid)

    # simple xgboost model
    # clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
    #                         subsample=0.8, nthread=10, learning_rate=0.2)

    # test with best params:
    clf = xgb.XGBClassifier(max_depth=5, n_estimators=200, colsample_bytree=1,gamma=1,min_child_weight=1,
                            subsample=0.6, nthread=10, learning_rate=0.2)


    clf.fit(xtrain_tfv.tocsc(), ytrain)


    predictions = clf.predict_proba(xvalid_tfv.tocsc())
    predictions_classes = clf.predict(xvalid_tfv.tocsc())

    print("xgboost tf-idf logloss: %0.3f " % multiclass_logloss(yvalid, predictions))  # 0.782
    print("xgboost tf-idf business friendly output: %0.3f" % (
            np.sum(predictions_classes == yvalid) / len(yvalid)))  # 0.665

    # simple xgboost on tf-idf svd features
    xtrain_svd, xvalid_svd = get_svd_features(xtrain_tfv, xvalid_tfv)

    clf.fit(xtrain_svd, ytrain)
    predictions = clf.predict_proba(xvalid_svd)
    predictions_classes = clf.predict(xvalid_svd)

    print("xgboost tf-idf+svd logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
    print("xgboost tf-idf+svd business friendly output: %0.3f" % (np.sum(predictions_classes == yvalid) / len(yvalid)))

    # grid-search model xgboost
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }

    mll_scorer = metrics.make_scorer(multiclass_logloss, greater_is_better=False, needs_proba=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)

    clf = xgb.XGBClassifier(n_estimators=200, nthread=1, learning_rate=0.02)
    #  brute-force grid search
    grid = GridSearchCV(estimator=clf, param_grid=params, scoring=mll_scorer, n_jobs=4,
                        cv=skf.split(xtrain_svd, ytrain),
                        verbose=3)
    grid.fit(xtrain_svd, ytrain)

    predictions = grid.best_estimator_.predict_proba(xvalid_svd)
    predictions_classes = grid.best_estimator_.peredict(xvalid_svd)

    # predictions = grid.predict_proba(xvalid_svd)
    # predictions_classes = grid.predict(xvalid_svd)

    print("rf grid xgboost (tf-ifd,svd) logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
    print("rf grid xgboost (tf-ifd,svd) business friendly output: %0.3f" % (
            np.sum(predictions_classes == yvalid) / len(yvalid)))
    print('\n Best parameters:')
    print(grid.best_params_)
