from nltk.corpus import stopwords
from sklearn import preprocessing, decomposition
# import xgboost as xgb
from sklearn.svm import SVC

from src.baseline_classifiers.lgr_tf_idf import *
from src.evaluations.evaluations import *
from src.features import tf_idf_features


def get_svd_features(xtrain, xtest):
    svd = decomposition.TruncatedSVD(n_components=120)  # up to 200 features to prevent long execution time...
    svd.fit(xtrain)
    xtrain_svd = svd.transform(xtrain)
    xvalid_svd = svd.transform(xtest)
    return xtrain_svd, xvalid_svd


if __name__ == '__main__':
    # LOAD DATA
    stop_words = stopwords.words('english')
    path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
    xtrain, xtest, ytrain, ytest = train_vali_split(train_df)

    # TF-IDF features
    print("TF-IDF + LOG REG")
    xtrain_tfv, xvalid_tfv = tf_idf_features.get_tfidf_word_features(xtrain, xtest)

    # Apply SVD, 120-200 components are good enough for SVM model.
    print("SVM + TFIDF")
    xtrain_svd ,xvalid_svd = get_svd_features(xtrain_tfv, xvalid_tfv)

    # Scale the data obtained from SVD.
    scl = preprocessing.StandardScaler()
    scl.fit(xtrain_svd)
    xtrain_svd_scl = scl.transform(xtrain_svd)
    xvalid_svd_scl = scl.transform(xvalid_svd)

    # Fitting a simple SVM
    clf = SVC(C=1.0, probability=True)  # since we need probabilities
    clf.fit(xtrain_svd_scl, ytrain)
    predictions = clf.predict_proba(xvalid_svd_scl)
    predictions_classes = clf.predict(xvalid_svd_scl)

    print("logloss: %0.3f " % multiclass_logloss(ytest, predictions))
    print("business friendly output: %0.3f" % (np.sum(predictions_classes == ytest) / len(ytest)))
