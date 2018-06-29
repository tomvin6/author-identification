from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from src.evaluations.logloss import multiclass_logloss
from src.utils.input_reader import *

# baseline-classifier
# Algorithm: logistic regression
# Features: TF-IDF

print("baseline-classifiers classifier")
print("Algorithm: logistic regression")
print("Features: TF-IDF")

# LOAD DATA
path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
xtrain, xtest, ytrain, ytest = train_vali_split(train_df)

# FEATURE CALCULATION
tfv = TfidfVectorizer(min_df=3, max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                      stop_words='english')
# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(xtrain) + list(xtest))  # Learn vocabulary and idf from training set.
# list of -> (# of sentence, occurred words number    tf-idf-score)
xtrain_tfv = tfv.transform(xtrain)  # create sparse matrix with tf-idf probs
xvalid_tfv = tfv.transform(xtest)

# Fitting a simple Logistic Regression on TFIDF
log_reg = LogisticRegression(C=1.0)
log_reg.fit(xtrain_tfv, ytrain)  # execute train for Log regression model

# ACCURACY & RESULTS
predictions = log_reg.predict_proba(xvalid_tfv)
preds = log_reg.predict(xvalid_tfv)
print("logloss: %0.3f " % multiclass_logloss(ytest, predictions))
print("business friendly output: %0.3f" % (np.sum(preds == ytest) / len(ytest)))
