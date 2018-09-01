import nltk
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

from src.baseline_classifiers.svm_tfidf import *
from src.features import fasttext_features
from src.features import probability_features
from src.features.writing_style_features import *
from sklearn.externals import joblib
from keras.models import load_model
import pickle as pickle
from sklearn.feature_extraction.text import TfidfTransformer

nltk.download('maxent_ne_chunker')
from src.features.santimant_features import *

# text to be processes should beunder 'text column
path_to_dumps = "xgboost_stacked_sub_mod_dumps"


# path_to_dumps = "tests"

# TODO- not tested!
def get_features_for_text(text_df):
    print("pre-process text...")
    text_df_processed = preprocess_text(pd.DataFrame(text_df, columns=['text']))
    print("finished pre-process!")

    print("adding stacked features...")
    print("1. on original text")
    loaded_vec = CountVectorizer(decode_error="replace",
                                 vocabulary=pickle.load(open(path_to_dumps + "\\vec_nb_orig_txt_ctv.pkl", "rb")))
    X = loaded_vec.transform(text_df_processed['text'])
    clf = joblib.load(path_to_dumps + "\\nb_orig_txt_ctv.pkl")
    prob_predictions = clf.predict_proba(X)
    for i in range(len(set(clf.predict(X)))):
        text_df_processed['nb_orig_txt_ctv' + str(i)] = prob_predictions[:, i]

    tfidftransformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",
                                 vocabulary=pickle.load(open(path_to_dumps + "\\vec_nb_orig_txt_char_tfidf.pkl", "rb")))
    X = tfidftransformer.fit_transform(loaded_vec.fit_transform(text_df_processed['text']))
    clf = joblib.load(path_to_dumps + "\\nb_orig_txt_char_tfidf.pkl")
    prob_predictions = clf.predict_proba(X)
    for i in range(len(set(clf.predict(X)))):
        text_df_processed['nb_orig_txt_char_tfidf' + str(i)] = prob_predictions[:, i]

    print("2. on lematized text")
    tfidftransformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",
                                 vocabulary=pickle.load(
                                     open(path_to_dumps + "\\vec_nb_txtcleaned_wrd_tfidf.pkl", "rb")))
    X = tfidftransformer.fit_transform(loaded_vec.fit_transform(text_df_processed['text_cleaned']))
    clf = joblib.load(path_to_dumps + "\\nb_txtcleaned_wrd_tfidf.pkl")
    prob_predictions = clf.predict_proba(X)
    for i in range(len(set(clf.predict(X)))):
        text_df_processed['nb_txtcleaned_wrd_tfidf' + str(i)] = prob_predictions[:, i]

    print("3. on entity annotated text")
    loaded_vec = CountVectorizer(decode_error="replace",
                                 vocabulary=pickle.load(open(path_to_dumps + "\\vec_nb_txtent_ctv.pkl", "rb")))
    X = loaded_vec.transform(text_df_processed['text_with_entities'])
    clf = joblib.load(path_to_dumps + "\\nb_txtent_ctv.pkl")
    prob_predictions = clf.predict_proba(X)
    for i in range(len(set(clf.predict(X)))):
        text_df_processed['nb_txtent_ctv' + str(i)] = prob_predictions[:, i]

    print("4. on pos-tagged text")
    loaded_vec = CountVectorizer(decode_error="replace",
                                 vocabulary=pickle.load(open(path_to_dumps + "\\vec_nb_txtpos_ctv.pkl", "rb")))
    X = loaded_vec.transform(text_df_processed['text_pos_tag_pairs'])
    clf = joblib.load(path_to_dumps + "\\nb_txtpos_ctv.pkl")
    prob_predictions = clf.predict_proba(X)
    for i in range(len(set(clf.predict(X)))):
        text_df_processed['nb_txtpos_ctv' + str(i)] = prob_predictions[:, i]

    print("finished stacked features additions!")

    print("adding fasttext features...")
    print("1. on original text")
    X, tokenizer = fasttext_features.create_docs(data=text_df_processed['text'], referance_col='text')
    fsx = load_model(path_to_dumps + "\\fsx_orgtxt_.h5")
    prob_predictions = fsx.predict_proba(X)
    for i in range(len(set(fsx.predict_classes(X)))):
        text_df_processed['fsx_orgtxt_' + str(i)] = prob_predictions[:, i]

    print("2. on lematized text")
    X, tokenizer = fasttext_features.create_docs(data=text_df_processed['text_cleaned'], referance_col='text_cleaned')
    fsx = load_model(path_to_dumps + "\\fsx_cleanedtxt_.h5")
    prob_predictions = fsx.predict_proba(X)
    for i in range(len(set(fsx.predict_classes(X)))):
        text_df_processed['fsx_cleanedtxt_' + str(i)] = prob_predictions[:, i]

    print("3. on entity annotated text")
    X, tokenizer = fasttext_features.create_docs(data=text_df_processed['text_with_entities'],
                                                 referance_col='text_with_entities')
    fsx = load_model(path_to_dumps + "\\fsx_enttxt_.h5")
    prob_predictions = fsx.predict_proba(X)
    for i in range(len(set(fsx.predict_classes(X)))):
        text_df_processed['fsx_enttxt_' + str(i)] = prob_predictions[:, i]

    print("4. on pos-tagged text")
    X, tokenizer = fasttext_features.create_docs(data=text_df_processed['text_with_entities'],
                                                 referance_col='text_pos_tag_pairs')
    fsx = load_model(path_to_dumps + "\\fsx_postxt_.h5")
    prob_predictions = fsx.predict_proba(X)
    for i in range(len(set(fsx.predict_classes(X)))):
        text_df_processed['fsx_postxt_' + str(i)] = prob_predictions[:, i]

    print("fast-text features added!")

    drop = ['text', 'text_cleaned', 'text_with_entities', 'text_pos_tag_pairs']
    text_df_processed = text_df_processed.drop(drop, axis=1)
    return text_df_processed


def train(train_df, preprocess=True):
    print("trainng xgb model with stacked features")

    if preprocess:
        print("Load data...")
        # path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
        # train_df, test_df, sample_df = load_data_sets(path_prefix + "train_short.csv", path_prefix + "test_short.csv", None)
        train_df = load_50_authors_data_sentences_to_dict()
        xtrain, xvalid, ytrain, yvalid = train_vali_split(train_df)
        print("data loaded!")
        print("pre-process text...")
        xtrain_processed = preprocess_text(pd.DataFrame(xtrain, columns=['text']))
        xvalid_processed = preprocess_text(pd.DataFrame(xvalid, columns=['text']))
        print("finished pre-process!")
    else:
        print("loading pre-process text...")
        xtrain_processed_df = load_50_authors_preprocessed_data()
        xtrain_processed, xvalid_processed, ytrain, yvalid = train_vali_split(xtrain_processed_df)

    xtrain_processed = xtrain_processed.reset_index(drop=True)
    xvalid_processed = xvalid_processed.reset_index(drop=True)

    print("adding stacked features...")
    print("1. on original text")
    # bag of words
    vectorizer = CountVectorizer(
        token_pattern=r'\w{1,}',
        ngram_range=(1, 2), stop_words='english'
    )
    model = probability_features.get_prob_vectorizer_features(xtrain_processed, xvalid_processed, ytrain, yvalid,
                                                              vectorizer, 'text', MultinomialNB(), 'nb_orig_txt_ctv',
                                                              cv=5)
    pickle.dump(vectorizer.vocabulary_, open(path_to_dumps + "\\vec_nb_orig_txt_ctv.pkl", "wb"))
    joblib.dump(model, path_to_dumps + "\\nb_orig_txt_ctv.pkl")

    # TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 5), analyzer='char'
    )
    model = probability_features.get_prob_vectorizer_features(xtrain_processed, xvalid_processed, ytrain, yvalid,
                                                              vectorizer, 'text',
                                                              MultinomialNB(), 'nb_orig_txt_char_tfidf', cv=5)
    pickle.dump(vectorizer.vocabulary_, open(path_to_dumps + "\\vec_nb_orig_txt_char_tfidf.pkl", "wb"))
    joblib.dump(model, path_to_dumps + "\\nb_orig_txt_char_tfidf.pkl")

    print("2. on lematized text")
    vectorizer = TfidfVectorizer(
        token_pattern=r'\w{1,}', ngram_range=(1, 1),
        use_idf=True, smooth_idf=True, sublinear_tf=True,
    )
    model = probability_features.get_prob_vectorizer_features(xtrain_processed, xvalid_processed, ytrain, yvalid,
                                                              vectorizer, 'text_cleaned', MultinomialNB(),
                                                              'nb_txtcleaned_wrd_tfidf', cv=5)
    pickle.dump(vectorizer.vocabulary_, open(path_to_dumps + "\\vec_nb_txtcleaned_wrd_tfidf.pkl", "wb"))
    joblib.dump(model, path_to_dumps + "\\nb_txtcleaned_wrd_tfidf.pkl")

    print("3. on entity annotated text")

    vectorizer = CountVectorizer(
        token_pattern=r'\w{1,}',
        ngram_range=(1, 2), stop_words='english'
    )
    model = probability_features.get_prob_vectorizer_features(xtrain_processed, xvalid_processed, ytrain, yvalid,
                                                              vectorizer, 'text_with_entities', MultinomialNB(),
                                                              'nb_txtent_ctv', cv=5)
    pickle.dump(vectorizer.vocabulary_, open(path_to_dumps + "\\vec_nb_txtent_ctv.pkl", "wb"))
    joblib.dump(model, path_to_dumps + "\\nb_txtent_ctv.pkl")

    print("4. on pos-tagged text")
    vectorizer = CountVectorizer(
        token_pattern=r'\w{1,}',
        ngram_range=(1, 2), stop_words='english'
    )
    model = probability_features.get_prob_vectorizer_features(xtrain_processed, xvalid_processed, ytrain, yvalid,
                                                              vectorizer, 'text_pos_tag_pairs', MultinomialNB(),
                                                              'nb_txtpos_ctv', cv=5)
    pickle.dump(vectorizer.vocabulary_, open(path_to_dumps + "\\vec_nb_txtpos_ctv.pkl", "wb"))
    joblib.dump(model, path_to_dumps + "\\nb_txtpos_ctv.pkl")

    print("finished stacked features additions!")

    print("adding fasttext features...")

    print("1. on original text")
    xtrain_fsx, xtest_fsx = fasttext_features.get_fasttext_features(xtrain_processed, ytrain, xvalid_processed, yvalid,
                                                                    referance_col='text',
                                                                    lbl_prefix='fsx_orgtxt_')
    xtrain_processed = pd.concat([xtrain_processed, xtrain_fsx], axis=1)
    xvalid_processed = pd.concat([xvalid_processed, xtest_fsx], axis=1)
    fsx = fasttext_features.obtain_fasttext_model(xtrain_processed, ytrain, xvalid_processed, yvalid,
                                                  referance_col='text')
    fsx.model.save(path_to_dumps + "\\fsx_orgtxt_.h5")

    print("2. on lematized text")
    xtrain_fsx, xtest_fsx = fasttext_features.get_fasttext_features(xtrain_processed, ytrain, xvalid_processed, yvalid,
                                                                    referance_col='text_cleaned',
                                                                    lbl_prefix='fsx_cleanedtxt_')
    xtrain_processed = pd.concat([xtrain_processed, xtrain_fsx], axis=1)
    xvalid_processed = pd.concat([xvalid_processed, xtest_fsx], axis=1)
    fsx = fasttext_features.obtain_fasttext_model(xtrain_processed, ytrain, xvalid_processed, yvalid,
                                                  referance_col='text_cleaned')
    fsx.model.save(path_to_dumps + "\\fsx_cleanedtxt_.h5")

    print("3. on entity annotated text")
    xtrain_fsx, xtest_fsx = fasttext_features.get_fasttext_features(xtrain_processed, ytrain, xvalid_processed, yvalid,
                                                                    referance_col='text_with_entities',
                                                                    lbl_prefix='fsx_enttxt_')
    xtrain_processed = pd.concat([xtrain_processed, xtrain_fsx], axis=1)
    xvalid_processed = pd.concat([xvalid_processed, xtest_fsx], axis=1)
    fsx = fasttext_features.obtain_fasttext_model(xtrain_processed, ytrain, xvalid_processed, yvalid,
                                                  referance_col='text_with_entities')
    fsx.model.save(path_to_dumps + "\\fsx_enttxt_.h5")

    print("4. on pos-tagged text")
    xtrain_fsx, xtest_fsx = fasttext_features.get_fasttext_features(xtrain_processed, ytrain, xvalid_processed, yvalid,
                                                                    referance_col='text_pos_tag_pairs',
                                                                    lbl_prefix='fsx_postxt_')
    xtrain_processed = pd.concat([xtrain_processed, xtrain_fsx], axis=1)
    xvalid_processed = pd.concat([xvalid_processed, xtest_fsx], axis=1)
    fsx = fasttext_features.obtain_fasttext_model(xtrain_processed, ytrain, xvalid_processed, yvalid,
                                                  referance_col='text_pos_tag_pairs')
    fsx.model.save(path_to_dumps + "\\fsx_postxt_.h5")

    print("fast-text features added!")

    drop = ['text', 'text_cleaned', 'text_with_entities', 'text_pos_tag_pairs']

    print("training xgboost model with all features...")
    print("features:")
    print(xtrain_processed.keys())
    X = xtrain_processed.drop(drop, axis=1)
    X_valid = xvalid_processed.drop(drop, axis=1)

    # xgbc = xgb.XGBClassifier(objective='multi:softprob', nthread=1)
    # xgb_par = {'min_child_weight': [1], 'colsample_bytree': [0.6], 'max_depth': [3],
    #            'subsample': [0.8], 'nthread': [-1], 'silent': [1]}
    #
    # grid_clf = GridSearchCV(xgbc, xgb_par, n_jobs=4, verbose=1, scoring='neg_log_loss', refit=True)
    # grid_clf.fit(X, ytrain);
    # print('LogLoss: %.3f' % grid_clf.best_score_)
    #
    # predictions = grid_clf.best_estimator_.predict_proba(X_valid)
    # predictions_classes = grid_clf.best_estimator_.predict(X_valid)
    #
    # print("xgboost accuracy: %0.3f" % (
    #         np.sum(predictions_classes == yvalid) / len(yvalid)))
    # print('\n Best parameters:')
    # xgb.plot_importance(grid_clf.best_estimator_)

    ########### build model ###########

    xgb_par = {'min_child_weight': 1, 'eta': 0.1, 'colsample_bytree': 0.3, 'max_depth': 3,
               'subsample': 0.8, 'lambda': 2.0, 'nthread': -1, 'silent': 1,
               'eval_metric': "mlogloss", 'objective': 'multi:softprob', 'num_class': 3}
    xtr = xgb.DMatrix(X, label=ytrain)
    xvl = xgb.DMatrix(X_valid, label=yvalid)

    watchlist = [(xtr, 'train'), (xvl, 'valid')]

    model_1 = xgb.train(xgb_par, xtr, 2000, watchlist, early_stopping_rounds=50,
                        maximize=False, verbose_eval=40)

    xgb.plot_importance(model_1)

    print('LogLoss: %.3f' % model_1.best_score)
    vl_prd = model_1.predict(xvl)
    vl_prd_cls = np.argmax(vl_prd, axis=1)
    print("Accuracy: %0.3f" % (np.sum(vl_prd_cls == yvalid) / len(yvalid)))


if __name__ == '__main__':
    print("get features for C50test data")
    test_df = load_50_authors_data_sentences_to_dict(train=False)
    get_features_for_text(test_df)
