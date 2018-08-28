from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from src.evaluations.evaluations import multiclass_logloss
from src.utils.input_reader import *
from src.features import tf_idf_features


# baseline-classifier
# Algorithm: logistic regression
# Features: TF-IDF

if __name__ == '__main__':
    print("baseline_classifiers classifier")
    print("Algorithm: logistic regression")
    print("Features: TF-IDF")

    # LOAD DATA
    path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
    xtrain, xtest, ytrain, ytest = train_vali_split(train_df)

    xtrain_tfv, xvalid_tfv = tf_idf_features.get_tfidf_word_features(xtrain, xtest)

    # Fitting a simple Logistic Regression on TFIDF
    log_reg = LogisticRegression(C=1.0)
    log_reg.fit(xtrain_tfv, ytrain)  # execute train for Log regression model

    # ACCURACY & RESULTS
    predictions = log_reg.predict_proba(xvalid_tfv)
    preds = log_reg.predict(xvalid_tfv)
    print("logloss: %0.3f " % multiclass_logloss(ytest, predictions))
    print("business friendly output: %0.3f" % (np.sum(preds == ytest) / len(ytest)))
