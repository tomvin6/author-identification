from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import sys
from src.data_analysis.statistics import load_50_auth_data
from src.features.writing_style_features import preprocess_text
from src.utils.input_reader import command_line_args, train_vali_split, load_50_authors_preprocessed_data
from sklearn import metrics
import numpy as np
import pandas as pd

def get_feature(xtrain, xtest):
    columns = ['punct_cnt', 'words_cnt', 'noun_chunks_cnt','ents_cnt', 'char_cnt','fraction_noun', 'fraction_adj', 'fraction_verbs','polarity_of_text','avg_wrd_ln']
    xtrain['char_cnt'] = xtrain['text'].apply(lambda row: len(str(row)))
    xtrain['avg_wrd_ln'] = xtrain['text'].apply(lambda row: avg_word_len(row))
    xtest['char_cnt'] = xtest['text'].apply(lambda row: len(str(row)))
    xtest['avg_wrd_ln'] = xtest['text'].apply(lambda row: avg_word_len(row))
    return xtrain[columns], xtest[columns]

def avg_word_len(sentence):
    words =sentence.split(' ')
    avg= sum(len(word) for word in words) / len(words)
    return avg

if __name__ == '__main__':

    print("baseline_classifiers classifier")
    print("Algorithm: Logistic regression on top of basic metadata features")
    # LOAD DATA
    # train_df = load_50_auth_data()
    train_df = load_50_authors_preprocessed_data()
    referance_col = 'text'
    if len(sys.argv) > 1:
        # command line args
        arg_dict = command_line_args(argv=sys.argv)

        if "file" in (arg_dict.keys()):
            input_data_path = str(arg_dict.get('file')[0])
            print("reading from external data file:" + input_data_path)
            df_train = pd.read_csv(input_data_path)
        if "preprocess" in (arg_dict.keys()):
            df_train = preprocess_text(df_train)
            if str(arg_dict.get('preprocess')[0]) == 'POS':
                referance_col = 'text_pos_tag_pairs'
            elif str(arg_dict.get('preprocess')[0]) == 'ENT':
                referance_col = 'text_with_entities'
            elif str(arg_dict.get('preprocess')[0]) == 'CLN':
                referance_col = 'text_cleaned'

    xtrain, xtest, ytrain, ytest = train_vali_split(train_df)
    # FEATURE CALCULATION- NB
    xtrain_feat, xvalid_feat = get_feature(xtrain, xtest)

    log_reg = LogisticRegression(C=1.0)
    log_reg.fit(xtrain_feat, ytrain)
    predictions_classes_lgr = log_reg.predict(xvalid_feat)
    predictions_lgr = log_reg.predict_proba(xvalid_feat)

    print("Logistic regression measures:")
    print("logloss: %0.3f " % metrics.log_loss(ytest, predictions_lgr))
    print("accuracy: %0.3f" % (np.sum(predictions_classes_lgr == ytest) / len(ytest)))

