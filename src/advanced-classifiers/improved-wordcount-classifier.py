from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.naive_bayes import MultinomialNB

from src.evaluations.logloss import multiclass_logloss
from src.utils.input_reader import *

# baseline-classifiers classifier
# Algorithm: logistic regression
# Features: TF-IDF

print("advance classifier")
print("Algorithm: Log regression")
print("Pre processing: removal of english stop words turned off, lowercase turn off")
print("Features: Word-count")

# LOAD DATA
path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
xtrain, xtest, ytrain, ytest = train_vali_split(train_df)

# PREPROCESSING DATA
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# STEM ALL WORDS IN TRAIN BEFORE ANALYSIS
'''joiner = lambda sentence_list: " ".join(sentence_list)
splitter = lambda sentence: sentence.split() 
stemmer = lambda sentence_list: [ps.stem(word) for word in sentence_list]
xtrain_stemmed = [joiner(stemmer(splitter(sentence))) for sentence in xtrain]
xtest_stemmed = [joiner(stemmer(splitter(sentence))) for sentence in xtest]
# enable stemming
xtrain = xtrain_stemmed
xtest = xtest_stemmed'''

# FEATURE CALCULATION
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), lowercase=False)

ctv.fit(list(xtrain) + list(xtest))
xtrain_ctv = ctv.transform(xtrain)
xvalid_ctv = ctv.transform(xtest)

# Fitting a simple Logistic Regression on Counts
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_ctv, ytrain)
predictions_classes = clf.predict(xvalid_ctv)
predictions = clf.predict_proba(xvalid_ctv)

print("logloss: %0.3f " % multiclass_logloss(ytest, predictions))
print("business friendly accuracy: %0.3f" % (np.sum(predictions_classes == ytest) / len(ytest)))