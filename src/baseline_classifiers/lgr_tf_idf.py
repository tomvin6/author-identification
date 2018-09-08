import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from src.data_analysis.statistics import load_50_auth_data
from src.evaluations.evaluations import multiclass_logloss
from src.features import tf_idf_features
from src.features.writing_style_features import preprocess_text
from src.utils.input_reader import *

# baseline-classifier
# Algorithm: logistic regression
# Features: TF-IDF

if __name__ == '__main__':
    print("baseline_classifiers classifier")
    print("Algorithm: logistic regression")
    print("Features: TF-IDF")

    # LOAD DATA
    # train_df = load_50_auth_data()
    train_df = load_50_authors_preprocessed_data()
    referance_col = 'text_cleaned'
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
            elif arg_dict.get('preprocess') == 'CLN':
                referance_col = 'text_cleaned'
        if "ngram" in (arg_dict.keys()):
            ngram = arg_dict.get('ngram')

    xtrain, xtest, ytrain, ytest = train_vali_split(train_df)
    xtrain = pd.DataFrame(xtrain[referance_col])
    xtrain = xtrain.rename(columns={referance_col: "text"})

    xtest = pd.DataFrame(xtest[referance_col])
    xtest = xtest.rename(columns={referance_col: "text"})

    xtrain_tfv, xvalid_tfv = tf_idf_features.get_tfidf_word_features(xtrain, xtest,ngram)

    # Fitting a simple Logistic Regression on TFIDF
    log_reg = LogisticRegression(C=1.0)
    log_reg.fit(xtrain_tfv, ytrain)  # execute train for Log regression model

    # ACCURACY & RESULTS
    predictions = log_reg.predict_proba(xvalid_tfv)
    preds = log_reg.predict(xvalid_tfv)
    print("logloss: %0.3f " % metrics.log_loss(ytest, predictions))
    print("accuracy: %0.3f" % (np.sum(preds == ytest) / len(ytest)))
