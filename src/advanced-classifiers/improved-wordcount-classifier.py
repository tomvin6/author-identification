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

# PREPROCESSING DATA - STEMMING (comment out, nnegative results effect)
'''
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
joiner = lambda sentence_list: " ".join(sentence_list)
splitter = lambda sentence: sentence.split() 
stemmer = lambda sentence_list: [ps.stem(word) for word in sentence_list]
xtrain_stemmed = [joiner(stemmer(splitter(sentence))) for sentence in xtrain]
xtest_stemmed = [joiner(stemmer(splitter(sentence))) for sentence in xtest]
# enable stemming
xtrain = xtrain_stemmed
xtest = xtest_stemmed
'''

# FEATURE CALCULATION
# bag of words
# creating a sparse matrix as BOW
# TODO: consider removing token pattern for better accuracy
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{0,}',
            ngram_range=(1, 2), lowercase=False, stop_words=None)

ctv.fit(list(xtrain) + list(xtest)) # calculate parameters: mean, dif
xtrain_ctv = ctv.transform(xtrain)  # create sparse BOW matrix
xvalid_ctv = ctv.transform(xtest)   # create sparse BOW matrix


# Fitting a simple Logistic Regression on Counts
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_ctv, ytrain)
predictions_classes = clf.predict(xvalid_ctv)
predictions = clf.predict_proba(xvalid_ctv)

print("logloss: %0.3f " % multiclass_logloss(ytest, predictions))
print("business friendly accuracy: %0.3f" % (np.sum(predictions_classes == ytest) / len(ytest)))

# making confusion matrix:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, predictions_classes)

# visualizing the training set results
'''
from matplotlib.colors import ListedColormap
x_set, y_set = xtrain_ctv, ytrain
X1, X2 = np.meshgrid(np.arange(start= x_set[:, 0].min() - 1, stop= x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start= x_set[:, 1].min() - 1, stop= x_set[:, 1].max() + 1, step = 0.01))
import matplotlib.pyplot as plt
plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.xlim(X2.min(), X2.max())
for i, j in enumerate(np.unique(ytrain)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1])
plt.title('logistic regression on t.s')
plt.legend()
plt.show()
'''