from sklearn.naive_bayes import MultinomialNB
import sys
from src.data_analysis.statistics import load_50_auth_data
from src.features.writing_style_features import preprocess_text
from src.utils.input_reader import command_line_args, train_vali_split, load_50_authors_preprocessed_data
from sklearn import metrics
import numpy as np

def get_feature(xtrain, xtest):
    columns = ['punct_cnt', 'words_cnt', 'noun_chunks_cnt', 'char_cnt', 'fraction_adj', 'fraction_verbs']
    xtrain['char_cnt'] = xtrain['text'].apply(lambda row: len(str(row)))
    xtest['char_cnt'] = xtest['text'].apply(lambda row: len(str(row)))
    return xtrain[columns], xtest[columns]


if __name__ == '__main__':

    print("baseline_classifiers classifier")
    print("Algorithm: NB on top of basic metadata features")
    # LOAD DATA
    # train_df = load_50_auth_data()
    train_df = load_50_authors_preprocessed_data()
    referance_col = 'text'
    plots = False
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
        if "plots" in (arg_dict.keys()):
            plots = arg_dict.get('plots')

    xtrain, xtest, ytrain, ytest = train_vali_split(train_df)
    # FEATURE CALCULATION- NB
    xtrain_feat, xvalid_feat = get_feature(xtrain, xtest)

    clfnb = MultinomialNB()
    clfnb.fit(xtrain_feat, ytrain)
    predictions_classes_nb = clfnb.predict(xvalid_feat)
    predictions_nb = clfnb.predict_proba(xvalid_feat)

    print("MultinomialNB measures:")
    print("logloss: %0.3f " % metrics.log_loss(ytest, predictions_nb))
    print("accuracy: %0.3f" % (np.sum(predictions_classes_nb == ytest) / len(ytest)))
