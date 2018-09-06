import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from src.baseline_classifiers.lgr_tf_idf import *
from src.data_analysis.statistics import load_50_auth_data
from src.evaluations.evaluations import multiclass_logloss
from src.features.writing_style_features import preprocess_text
from src.utils.input_reader import *

# baseline_classifiers classifier
# Algorithm: MultinomialNB
# Features: TF-IDF
if __name__ == '__main__':
    print("baseline_classifiers classifier")
    print("Algorithm: Naive bayes")
    print("Features: TF-IDF")

    # LOAD DATA
    train_df = load_50_auth_data()
    # train_df = load_50_authors_preprocessed_data()
    referance_col = 'text'
    ngram = 1
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

    # FEATURE CALCULATION
    xtrain_tfv, xvalid_tfv = tf_idf_features.get_tfidf_word_features(xtrain, xtest, ngram)

    # Fitting a simple NB on TFIDF
    clf = MultinomialNB()
    clf.fit(xtrain_tfv, ytrain)

    # ACCURACY & RESULTS
    predictions = clf.predict_proba(xvalid_tfv)
    predictions_classes = clf.predict(xvalid_tfv)
    print("logloss: %0.3f " % metrics.log_loss(ytest, predictions))
    print("accuracy: %0.3f" % (np.sum(predictions_classes == ytest) / len(ytest)))
