from sklearn.naive_bayes import MultinomialNB

from src.baseline_classifiers.tf_idf import *
from src.evaluations.logloss import multiclass_logloss
from src.utils.input_reader import *

# baseline_classifiers classifier
# Algorithm: logistic regression
# Features: TF-IDF
if __name__ == '__main__':
    print("baseline_classifiers classifier")
    print("Algorithm: Naive bayes")
    print("Features: TF-IDF")

    # LOAD DATA
    path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
    xtrain, xtest, ytrain, ytest = train_vali_split(train_df)

    # FEATURE CALCULATION
    xtrain_tfv, xvalid_tfv = get_dfidf_features(xtrain, xtest)

    # Fitting a simple Logistic Regression on TFIDF
    clf = MultinomialNB()
    clf.fit(xtrain_tfv, ytrain)

    # ACCURACY & RESULTS
    predictions = clf.predict_proba(xvalid_tfv)
    predictions_classes = clf.predict(xvalid_tfv)
    print("logloss: %0.3f " % multiclass_logloss(ytest, predictions))
    print("business friendly accuracy: %0.3f" % (np.sum(predictions_classes == ytest) / len(ytest)))
