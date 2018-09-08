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
    train_df = load_50_auth_data()
    # train_df = load_50_authors_preprocessed_data()
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

    log_reg = LogisticRegression(C=1.0)
    log_reg.fit(xtrain_feat, ytrain)
    predictions_classes_lgr = log_reg.predict(xvalid_feat)
    predictions_lgr = log_reg.predict_proba(xvalid_feat)

    print("Logistic regression measures:")
    print("logloss: %0.3f " % metrics.log_loss(ytest, predictions_lgr))
    print("accuracy: %0.3f" % (np.sum(predictions_classes_lgr == ytest) / len(ytest)))

    # xgb_par = {'min_child_weight': 1, 'eta': 0.1, 'colsample_bytree': 0.3, 'max_depth': 3,
    #            'subsample': 0.8, 'lambda': 2.0, 'nthread': -1, 'silent': 1,
    #            'eval_metric': "mlogloss", 'objective': 'multi:softprob', 'num_class': len(set(ytrain))}
    # xtr = xgb.DMatrix(xtrain_feat, label=ytrain)
    # xvl = xgb.DMatrix(xvalid_feat, label=ytest)
    #
    # watchlist = [(xtr, 'train'), (xvl, 'valid')]
    #
    # model_1 = xgb.train(xgb_par, xtr, 2000, watchlist, early_stopping_rounds=50,
    #                     maximize=False, verbose_eval=40)
    # print('LogLoss: %.3f' % model_1.best_score)
    # vl_prd = model_1.predict(xvl)
    # vl_prd_cls = np.argmax(vl_prd, axis=1)
    # print("Accuracy: %0.3f" % (np.sum(vl_prd_cls == yvalid) / len(yvalid)))


