from nltk.corpus import stopwords
from sklearn import preprocessing, decomposition
# import xgboost as xgb
from sklearn.svm import SVC
from sklearn import metrics
from src.baseline_classifiers.lgr_tf_idf import *
from src.evaluations.evaluations import *
from src.features import tf_idf_features


def get_svd_features(xtrain, xtest):
    svd = decomposition.TruncatedSVD(n_components=200)  # up to 200 features to prevent long execution time...
    svd.fit(xtrain)
    xtrain_svd = svd.transform(xtrain)
    xvalid_svd = svd.transform(xtest)
    return xtrain_svd, xvalid_svd


if __name__ == '__main__':
    print("baseline_classifiers classifier")
    print("Algorithm: SVM on top of TF-IDF features, with svd feature reduction")
    # LOAD DATA
    # train_df = load_50_auth_data()
    train_df = load_50_authors_preprocessed_data()
    ngram = 3
    referance_col = 'text_pos_tag_pairs'
    if len(sys.argv) > 1:
        # command line args
        arg_dict = command_line_args(argv=sys.argv)

        if "file" in (arg_dict.keys()):
            input_data_path = arg_dict.get('file')
            print("reading from external data file:" + input_data_path)
            df_train = pd.read_csv(input_data_path)
        if "preprocess" in (arg_dict.keys()):
            df_train = preprocess_text(df_train)
            if arg_dict.get('preprocess') == 'POS':
                referance_col = 'text_pos_tag_pairs'
            elif arg_dict.get('preprocess') == 'ENT':
                referance_col = 'text_with_entities'
            elif arg_dict.get('preprocess') == 'ENT':
                referance_col = 'text_cleaned'
        if "ngram" in (arg_dict.keys()):
            ngram = arg_dict.get('ngram')

    xtrain, xtest, ytrain, ytest = train_vali_split(train_df)
    xtrain = pd.DataFrame(xtrain[referance_col])
    xtrain = xtrain.rename(columns={referance_col: "text"})

    xtest = pd.DataFrame(xtest[referance_col])
    xtest = xtest.rename(columns={referance_col: "text"})

    # TF-IDF features
    print("TF-IDF features")
    xtrain_tfv, xvalid_tfv = tf_idf_features.get_tfidf_word_features(xtrain, xtest, ngram)

    # Apply SVD, 120-200 components are good enough for SVM model.
    print("SVD features reduction (200 features)")
    xtrain_svd, xvalid_svd = get_svd_features(xtrain_tfv, xvalid_tfv)

    # Scale the data obtained from SVD.
    scl = preprocessing.StandardScaler()
    scl.fit(xtrain_svd)
    xtrain_svd_scl = scl.transform(xtrain_svd)
    xvalid_svd_scl = scl.transform(xvalid_svd)

    # Fitting a simple SVM
    clf = SVC(C=1.0, probability=True)
    clf.fit(xtrain_svd_scl, ytrain)
    predictions = clf.predict_proba(xvalid_svd_scl)
    predictions_classes = clf.predict(xvalid_svd_scl)

    print("logloss: %0.3f " % metrics.log_loss(ytest, predictions))
    print("business friendly output: %0.3f" % (np.sum(predictions_classes == ytest) / len(ytest)))
