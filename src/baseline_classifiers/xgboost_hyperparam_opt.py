from src.baseline_classifiers.svm_tfidf import *
import xgboost as xgb
from nltk.corpus import stopwords
# import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

from src.features import tf_idf_features
from src.baseline_classifiers.svm_tfidf import *
from src.evaluations.logloss import *

from src.features import tf_idf_features
from src.features import naive_bayes_fetures
from src.features import svd_features
from src.features import writing_style_features
import pandas as pd


def run_xgboost_random_search():
    stop_words = stopwords.words('english')
    print("baseline_classifiers classifier")
    print("Algorithm: XGboost with hyperparam optimization (grid search) ")
    print("Features: TF-IDF")

    # LOAD DATA
    path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
    xtrain, xvalid, ytrain, yvalid = train_vali_split(train_df)

    xtrain_tfv, xvalid_tfv = tf_idf_features.get_dfidf_features(xtrain, xvalid)

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


def run_xgboost_grid_search():
    stop_words = stopwords.words('english')
    print("baseline_classifiers classifier")
    print("Algorithm: XGboost with hyperparam optimization (grid search) ")
    print("Features: TF-IDF")

    # LOAD DATA
    path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
    xtrain, xvalid, ytrain, yvalid = train_vali_split(train_df)

    xtrain_tfv, xvalid_tfv = tf_idf_features.get_dfidf_features(xtrain, xvalid)

    # simple xgboost model
    # clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
    #                         subsample=0.8, nthread=10, learning_rate=0.2)

    # test with best params:
    clf = xgb.XGBClassifier(max_depth=5, n_estimators=200, colsample_bytree=1, gamma=1, min_child_weight=1,
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


def test_features():
    # read input data
    path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
    xtrain, xvalid, ytrain, yvalid = train_vali_split(train_df)

    ########### collect features ###########
    # writing style features
    xtrain = writing_style_features.get_writing_style_features(xtrain)
    xvalid = writing_style_features.get_writing_style_features(xvalid)
    xtrain.keys()

    # TF-IDF
    xtrain_tfidf_wrd, xtest_tfidf_wrd = tf_idf_features.get_tfidf_word_features(xtrain, xvalid)
    xtrain_tfidf_chr, xtest_tfidf_chr = tf_idf_features.get_tfidf_char_features(xtrain, xvalid)

    # Naeive-Bayes
    xtrain_nb_wrd, xtest_nb_wrd = naive_bayes_fetures.get_nb_features(xtrain_tfidf_wrd, ytrain, xtest_tfidf_wrd,
                                                                      'nb_wrd')
    xtrain_nb_chr, xtest_nb_chr = naive_bayes_fetures.get_nb_features(xtrain_tfidf_wrd, ytrain, xtest_tfidf_wrd,
                                                                      'nb_chr')
    xtrain = pd.concat([xtrain, xtrain_nb_wrd, xtrain_nb_chr], axis=1)
    xvalid = pd.concat([xvalid, xtest_nb_wrd, xtest_nb_chr], axis=1)
    del xtrain_nb_wrd, xtest_nb_wrd, xtrain_nb_chr, xtest_nb_chr

    # SVD
    xtrain_svd_wrd, xtest_svd_wrd = svd_features.get_svd_features(xtrain_tfidf_wrd, xtest_tfidf_wrd, 'svd_wrd_')
    xtrain_svd_chr, xtest_svd_chr = svd_features.get_svd_features(xtrain_tfidf_chr, xtest_tfidf_chr, 'svd_chr_')
    xtrain = pd.concat([xtrain, xtrain_svd_wrd, xtrain_svd_chr], axis=1)
    xvalid = pd.concat([xvalid, xtest_svd_wrd, xtest_svd_chr], axis=1)
    del xtrain_svd_wrd, xtest_svd_wrd, xtrain_svd_chr, xtest_svd_chr

    ########### build model ###########

    xgb_par = {'min_child_weight': 1, 'eta': 0.1, 'colsample_bytree': 0.7, 'max_depth': 3,
               'subsample': 0.8, 'lambda': 2.0, 'nthread': -1, 'silent': 1,
               'eval_metric': "mlogloss", 'objective': 'multi:softprob', 'num_class': 3}
    xtr = xgb.DMatrix(xtrain.drop(['text'], axis=1), label=ytrain)
    xvl = xgb.DMatrix(xvalid.drop(['text'], axis=1), label=yvalid)

    watchlist = watchlist = [(xtr, 'train'), (xvl, 'valid')]

    model_1 = xgb.train(xgb_par, xtr, 1000, watchlist, early_stopping_rounds=50,
                        maximize=False, verbose_eval=40)
    print('Modeling RMSLE %.5f' % model_1.best_score)

    return


if __name__ == '__main__':
    test_features()
