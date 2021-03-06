import pickle as pickle

import xgboost as xgb
from keras.models import load_model
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.baseline_classifiers.svm_tfidf import *
from src.features import fasttext_features
from src.features import probability_features
from src.features.writing_style_features import *
from src.utils.confusion import *
import pandas as pd
nltk.download('maxent_ne_chunker')
from src.features.santimant_features import *
import matplotlib.pyplot as plt

# text to be processes should be under 'text column
path_to_dumps = "xgboost_stacked_sub_mod_dumps"
root = ".." + os.sep + ".." + os.sep + 'src' + os.sep + 'baseline_classifiers' + os.sep + path_to_dumps


def get_features_for_text(text_df, preprocess=True):
    if preprocess:
        print("pre-process text...")
        text_df_processed = preprocess_text(pd.DataFrame(text_df, columns=['text']))
        print("finished pre-process!")
    else:
        text_df_processed = text_df

    print("adding stacked features...")
    print("1. on original text")
    loaded_vec = CountVectorizer(decode_error="replace",
                                 vocabulary=pickle.load(open(root + "\\vec_nb_orig_txt_ctv.pkl", "rb")))
    X = loaded_vec.transform(text_df_processed['text'])
    clf = joblib.load(root + "\\nb_orig_txt_ctv.pkl")
    prob_predictions = clf.predict_proba(X)
    for i in range(prob_predictions.shape[1]):
        text_df_processed['nb_orig_txt_ctv' + str(i)] = prob_predictions[:, i]

    tfidftransformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",
                                 vocabulary=pickle.load(open(root + "\\vec_nb_orig_txt_char_tfidf.pkl", "rb")))
    X = tfidftransformer.fit_transform(loaded_vec.fit_transform(text_df_processed['text']))
    clf = joblib.load(root + "\\nb_orig_txt_char_tfidf.pkl")
    prob_predictions = clf.predict_proba(X)
    for i in range(prob_predictions.shape[1]):
        text_df_processed['nb_orig_txt_char_tfidf' + str(i)] = prob_predictions[:, i]

    print("2. on lematized text")
    tfidftransformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",
                                 vocabulary=pickle.load(
                                     open(root + "\\vec_nb_txtcleaned_wrd_tfidf.pkl", "rb")))
    X = tfidftransformer.fit_transform(loaded_vec.fit_transform(text_df_processed['text_cleaned']))
    clf = joblib.load(root + "\\nb_txtcleaned_wrd_tfidf.pkl")
    prob_predictions = clf.predict_proba(X)
    for i in range(prob_predictions.shape[1]):
        text_df_processed['nb_txtcleaned_wrd_tfidf' + str(i)] = prob_predictions[:, i]

    print("3. on entity annotated text")
    loaded_vec = CountVectorizer(decode_error="replace",
                                 vocabulary=pickle.load(open(root + "\\vec_nb_txtent_ctv.pkl", "rb")))
    X = loaded_vec.transform(text_df_processed['text_with_entities'])
    clf = joblib.load(root + "\\nb_txtent_ctv.pkl")
    prob_predictions = clf.predict_proba(X)
    for i in range(prob_predictions.shape[1]):
        text_df_processed['nb_txtent_ctv' + str(i)] = prob_predictions[:, i]

    print("4. on pos-tagged text")
    loaded_vec = CountVectorizer(decode_error="replace",
                                 vocabulary=pickle.load(open(root + "\\vec_nb_txtpos_ctv.pkl", "rb")))
    X = loaded_vec.transform(text_df_processed['text_pos_tag_pairs'])
    clf = joblib.load(root + "\\nb_txtpos_ctv.pkl")
    prob_predictions = clf.predict_proba(X)
    for i in range(prob_predictions.shape[1]):
        text_df_processed['nb_txtpos_ctv' + str(i)] = prob_predictions[:, i]

    print("finished stacked features additions!")

    print("adding fasttext features...")
    print("1. on original text")
    loaded_tokenizer = pickle.load(open(root + "\\vec_fsx_orgtxt_.pkl", "rb"))
    X = fasttext_features.create_docs(data=text_df_processed['text'], train_mode=False, tokenizer=loaded_tokenizer,
                                      referance_col='text')
    fsx = load_model(root + "\\fsx_orgtxt_.h5")
    prob_predictions = fsx.predict_proba(X)
    for i in range(prob_predictions.shape[1]):
        text_df_processed['fsx_orgtxt_' + str(i)] = prob_predictions[:, i]

    print("2. on lematized text")
    loaded_tokenizer = pickle.load(open(root + "\\vec_fsx_cleanedtxt_.pkl", "rb"))
    X = fasttext_features.create_docs(data=text_df_processed['text_cleaned'], train_mode=False,
                                      tokenizer=loaded_tokenizer, referance_col='text_cleaned')
    fsx = load_model(root + "\\fsx_cleanedtxt_.h5")
    prob_predictions = fsx.predict_proba(X)
    for i in range(prob_predictions.shape[1]):
        text_df_processed['fsx_cleanedtxt_' + str(i)] = prob_predictions[:, i]

    print("3. on entity annotated text")
    loaded_tokenizer = pickle.load(open(root + "\\vec_fsx_enttxt_.pkl", "rb"))
    X = fasttext_features.create_docs(data=text_df_processed['text_with_entities'], train_mode=False,
                                      tokenizer=loaded_tokenizer,
                                      referance_col='text_with_entities')
    fsx = load_model(root + "\\fsx_enttxt_.h5")
    prob_predictions = fsx.predict_proba(X)
    for i in range(prob_predictions.shape[1]):
        text_df_processed['fsx_enttxt_' + str(i)] = prob_predictions[:, i]

    print("4. on pos-tagged text")
    loaded_tokenizer = pickle.load(open(root + "\\vec_fsx_postxt_.pkl", "rb"))
    X = fasttext_features.create_docs(data=text_df_processed['text_pos_tag_pairs'], train_mode=False,
                                      tokenizer=loaded_tokenizer,
                                      referance_col='text_pos_tag_pairs')
    fsx = load_model(root + "\\fsx_postxt_.h5")
    prob_predictions = fsx.predict_proba(X)
    for i in range(prob_predictions.shape[1]):
        text_df_processed['fsx_postxt_' + str(i)] = prob_predictions[:, i]

    print("fast-text features added!")
    drop = ['text', 'text_cleaned', 'text_with_entities', 'text_pos_tag_pairs', 'idx', 'author', 'author_label', 'id']
    text_df_processed = text_df_processed.drop(drop, axis=1)

    print("adding gbm answer for data")
    cls = joblib.load(root + "\\xgboost_model.h5")
    mtx_text_df_processed = xgb.DMatrix(text_df_processed)
    prob_predictions = cls.predict(mtx_text_df_processed)
    for i in range(prob_predictions.shape[1]):
        text_df_processed['xgboost_' + str(i)] = prob_predictions[:, i]
    xgb_ans = pd.DataFrame()
    for i in range(prob_predictions.shape[1]):
        xgb_ans['xgboost_' + str(i)] = prob_predictions[:, i]

    print("gbm answers added!")
    return text_df_processed, xgb_ans


def train(train_df=None, preprocess=True, sentences=True):
    print("trainng xgb model with stacked features")

    # if preprocess:
    #     print("Load data...")
    #     # path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    #     # train_df, test_df, sample_df = load_data_sets(path_prefix + "train_short.csv", path_prefix + "test_short.csv", None)
    #     if sentences:
    #         train_df = load_50_authors_data_sentences_to_dict()
    #     else:
    #         train_df= load_50_authors_data_sets_to_dict()
    #     xtrain, xvalid, ytrain, yvalid = train_vali_split(train_df)
    #     print("data loaded!")
    #     print("pre-process text...")
    #     xtrain_processed = preprocess_text(pd.DataFrame(xtrain, columns=['text']))
    #     xvalid_processed = preprocess_text(pd.DataFrame(xvalid, columns=['text']))
    #     print("finished pre-process!")
    # else:
    #     print("loading pre-process text...")
    #     xtrain_processed_df = load_50_authors_preprocessed_data(sentences)
    #     xtrain_processed, xvalid_processed, ytrain, yvalid = train_vali_split(xtrain_processed_df)

    xtrain_processed, xvalid_processed, ytrain, yvalid = train_vali_split(train_df)
    xtrain_processed = xtrain_processed.reset_index(drop=True)
    xvalid_processed = xvalid_processed.reset_index(drop=True)

    print("adding stacked features...")
    print("1. on original text")
    # bag of words
    vectorizer = CountVectorizer(
        token_pattern=r'\w{1,}',
        ngram_range=(1, 3), stop_words='english'
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
    print("saved features files:")
    print(path_to_dumps + "\\nb_orig_txt_ctv.pkl")
    print(path_to_dumps + "\\nb_orig_txt_char_tfidf.pkl")

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
    print("saved features files:")
    print(path_to_dumps + "\\nb_txtcleaned_wrd_tfidf.pkl")

    print("3. on entity annotated text")

    vectorizer = CountVectorizer(
        token_pattern=r'\w{1,}',
        ngram_range=(1, 3), stop_words='english'
    )
    model = probability_features.get_prob_vectorizer_features(xtrain_processed, xvalid_processed, ytrain, yvalid,
                                                              vectorizer, 'text_with_entities', MultinomialNB(),
                                                              'nb_txtent_ctv', cv=5)
    pickle.dump(vectorizer.vocabulary_, open(path_to_dumps + "\\vec_nb_txtent_ctv.pkl", "wb"))
    joblib.dump(model, path_to_dumps + "\\nb_txtent_ctv.pkl")

    print("saved features files:")
    print(path_to_dumps + "\\nb_txtent_ctv.pkl")

    print("4. on pos-tagged text")
    vectorizer = CountVectorizer(
        token_pattern=r'\w{1,}',
        ngram_range=(1, 3), stop_words='english'
    )
    model = probability_features.get_prob_vectorizer_features(xtrain_processed, xvalid_processed, ytrain, yvalid,
                                                              vectorizer, 'text_pos_tag_pairs', MultinomialNB(),
                                                              'nb_txtpos_ctv', cv=5)
    pickle.dump(vectorizer.vocabulary_, open(path_to_dumps + "\\vec_nb_txtpos_ctv.pkl", "wb"))
    joblib.dump(model, path_to_dumps + "\\nb_txtpos_ctv.pkl")

    print("saved features files:")
    print(path_to_dumps + "\\nb_txtpos_ctv.pkl")

    print("finished stacked features additions!")

    print("adding fasttext features...")

    print("1. on original text")
    xtrain_fsx, xtest_fsx = fasttext_features.get_fasttext_features(xtrain_processed, ytrain, xvalid_processed, yvalid,
                                                                    referance_col='text',
                                                                    lbl_prefix='fsx_orgtxt_')
    xtrain_processed = pd.concat([xtrain_processed, xtrain_fsx], axis=1)
    xvalid_processed = pd.concat([xvalid_processed, xtest_fsx], axis=1)
    fsx, tokenizer = fasttext_features.obtain_fasttext_model(xtrain_processed, ytrain, xvalid_processed, yvalid,
                                                             referance_col='text')

    pickle.dump(tokenizer, open(path_to_dumps + "\\vec_fsx_orgtxt_.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    fsx.model.save(path_to_dumps + "\\fsx_orgtxt_.h5")

    print("saved features files:")
    print(path_to_dumps + "\\fsx_orgtxt_.h5")

    print("2. on lematized text")
    xtrain_fsx, xtest_fsx = fasttext_features.get_fasttext_features(xtrain_processed, ytrain, xvalid_processed, yvalid,
                                                                    referance_col='text_cleaned',
                                                                    lbl_prefix='fsx_cleanedtxt_')
    xtrain_processed = pd.concat([xtrain_processed, xtrain_fsx], axis=1)
    xvalid_processed = pd.concat([xvalid_processed, xtest_fsx], axis=1)
    fsx, tokenizer = fasttext_features.obtain_fasttext_model(xtrain_processed, ytrain, xvalid_processed, yvalid,
                                                             referance_col='text_cleaned')
    pickle.dump(tokenizer, open(path_to_dumps + "\\vec_fsx_cleanedtxt_.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    fsx.model.save(path_to_dumps + "\\fsx_cleanedtxt_.h5")

    print("saved features files:")
    print(path_to_dumps + "\\fsx_cleanedtxt_.h5")

    print("3. on entity annotated text")
    xtrain_fsx, xtest_fsx = fasttext_features.get_fasttext_features(xtrain_processed, ytrain, xvalid_processed, yvalid,
                                                                    referance_col='text_with_entities',
                                                                    lbl_prefix='fsx_enttxt_')
    xtrain_processed = pd.concat([xtrain_processed, xtrain_fsx], axis=1)
    xvalid_processed = pd.concat([xvalid_processed, xtest_fsx], axis=1)
    fsx, tokenizer = fasttext_features.obtain_fasttext_model(xtrain_processed, ytrain, xvalid_processed, yvalid,
                                                             referance_col='text_with_entities')

    pickle.dump(tokenizer, open(path_to_dumps + "\\vec_fsx_enttxt_.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    fsx.model.save(path_to_dumps + "\\fsx_enttxt_.h5")

    print("saved features files:")
    print(path_to_dumps + "\\fsx_enttxt_.h5")

    print("4. on pos-tagged text")
    xtrain_fsx, xtest_fsx = fasttext_features.get_fasttext_features(xtrain_processed, ytrain, xvalid_processed, yvalid,
                                                                    referance_col='text_pos_tag_pairs',
                                                                    lbl_prefix='fsx_postxt_')
    xtrain_processed = pd.concat([xtrain_processed, xtrain_fsx], axis=1)
    xvalid_processed = pd.concat([xvalid_processed, xtest_fsx], axis=1)
    fsx, tokenizer = fasttext_features.obtain_fasttext_model(xtrain_processed, ytrain, xvalid_processed, yvalid,
                                                             referance_col='text_pos_tag_pairs')
    pickle.dump(tokenizer, open(path_to_dumps + "\\vec_fsx_postxt_.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    fsx.model.save(path_to_dumps + "\\fsx_postxt_.h5")

    print("saved features files:")
    print(path_to_dumps + "\\fsx_postxt_.h5")

    print("fast-text features added!")

    drop = ['text', 'text_cleaned', 'text_with_entities', 'text_pos_tag_pairs']

    print("training xgboost model with all features...")
    print("features:")
    X = xtrain_processed.drop(drop, axis=1)
    print(X.keys())
    X_valid = xvalid_processed.drop(drop, axis=1)

    ########### build model ###########

    xgb_par = {'min_child_weight': 1, 'eta': 0.1, 'colsample_bytree': 0.3, 'max_depth': 3,
               'subsample': 0.8, 'lambda': 2.0, 'nthread': -1, 'silent': 1,
               'eval_metric': "mlogloss", 'objective': 'multi:softprob', 'num_class': len(set(ytrain))}
    xtr = xgb.DMatrix(X, label=ytrain)
    xvl = xgb.DMatrix(X_valid, label=yvalid)

    watchlist = [(xtr, 'train'), (xvl, 'valid')]

    model_1 = xgb.train(xgb_par, xtr, 2000, watchlist, early_stopping_rounds=50,
                        maximize=False, verbose_eval=40)

    print('LogLoss: %.3f' % model_1.best_score)
    vl_prd = model_1.predict(xvl)
    vl_prd_cls = np.argmax(vl_prd, axis=1)
    print("Accuracy: %0.3f" % (np.sum(vl_prd_cls == yvalid) / len(yvalid)))

    # plot importance
    f_scores = model_1.get_fscore()
    sorted_f_scores = sorted(f_scores.items(), reverse=True, key=lambda x: x[1])
    top_f_scores = sorted_f_scores[:30]
    feature = []
    f_score = []
    for i in range(0, len(top_f_scores)):
        str = top_f_scores[i][0]
        val = top_f_scores[i][1]
        feature.append(str)
        f_score.append(val)

    fig = plt.figure()
    plt.plot(f_score, feature)
    plt.title('top 30 features F-scores')
    plt.xlabel('F-score')
    plt.ylabel('feature name')
    fig.tight_layout()
    fig.savefig('suppervised_feature_importance_sentencesdb.pdf', format='pdf')

    joblib.dump(model_1, path_to_dumps + "\\xgboost_model.h5")
    print("saved xgboost model files:")
    print(path_to_dumps + "\\xgboost_model.h5")

    # plot confusion matrix
    labels = list(set(ytrain))
    cnf_matrix = confusion_matrix(yvalid, vl_prd_cls)
    np.set_printoptions(precision=2)
    fig = plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels,
                          title='Confusion matrix', normalize=True)
    fig.tight_layout()
    fig.savefig('confusion_xgboost.pdf', format='pdf')


if __name__ == '__main__':
    # print("get features for C50test data")
    # test_df = load_50_authors_preprocessed_data(train=False)
    # get_features_for_text(test_df)
    print("baseline_classifiers xgboost stacked classifier")
    print("Algorithm: GBM on top of multiple features")

    # print("train xgboost model")
    # train(preprocess=False)

    # defule- load 50 authors data as train
    # df = load_50_authors_preprocessed_data()
    df = load_50_authors_preprocessed_data()
    train_yn = True
    preprocess = False
    output_data_path = "output_predictions_sentences.tsv"

    if len(sys.argv) > 1:
        # command line args
        arg_dict = command_line_args(argv=sys.argv)
        if "file" in (arg_dict.keys()):
            input_data_path = str(arg_dict.get('file')[0])
            print("reading from external data file:" + input_data_path)
            df = pd.read_csv(input_data_path)
        if "output_file" in (arg_dict.keys()):
            output_data_path = str(arg_dict.get('output_file')[0])
            print("output data file:" + output_data_path)
        if "train" in (arg_dict.keys()):
            train_yn = arg_dict.get('train')[0]
            if train_yn == 'False':
                train_yn = False
            else:
                train_yn = True
            print("train mode:" + str(train_yn))
        if "preprocess" in (arg_dict.keys()):
            preprocess = arg_dict.get('preprocess')[0]
            if preprocess == 'False':
                preprocess = False
            else:
                preprocess = True

    if preprocess:
        df = preprocess_text(df)

    if train_yn:
        train(df)
    else:
        feat_df,xgb_ans = get_features_for_text(df, preprocess=preprocess)
        feat_df.to_csv(output_data_path, sep='\t')
