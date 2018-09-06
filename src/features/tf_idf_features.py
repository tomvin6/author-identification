from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd


# get tf-idf features on a given trainning and validation sets
# input df's are with a single column on top of ot the tf-idf features will be added
def get_tfidf_word_features(xtrain, xvalid, ngram=3):
    tfv = TfidfVectorizer(min_df=3, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, ngram), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')
    # Fitting TF-IDF to both training and test sets (semi-supervised learning)
    tfv.fit_transform(list(xtrain['text']) + list(xvalid['text']))  # Learn vocabulary and idf from training set.
    # list of -> (# of sentence, occurred words number    tf-idf-score)
    xtrain_tfv = tfv.transform(xtrain['text'])  # create sparse matrix with tf-idf probs
    xvalid_tfv = tfv.transform(xvalid['text'])
    return xtrain_tfv, xvalid_tfv


def get_tfidf_char_features(xtrain, xvalid,ngram=3):
    tfv = TfidfVectorizer(min_df=3, max_features=None,
                          strip_accents='unicode', analyzer='char', token_pattern=r'\w{1,}',
                          ngram_range=(1, ngram), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')
    # Fitting TF-IDF to both training and test sets (semi-supervised learning)
    tfv.fit_transform(list(xtrain['text']) + list(xvalid['text']))  # Learn vocabulary and idf from training set.
    # list of -> (# of sentence, occurred words number    tf-idf-score)
    xtrain_tfv = tfv.transform(xtrain['text'])  # create sparse matrix with tf-idf probs
    xvalid_tfv = tfv.transform(xvalid['text'])
    return xtrain_tfv, xvalid_tfv


def get_count_features(xtrain, xvalid):
    tfv = CountVectorizer(ngram_range=(1, 3), lowercase=False, stop_words=None)
    # Fitting TF-IDF to both training and test sets (semi-supervised learning)
    tfv.fit_transform(list(xtrain['text']) + list(xvalid['text']))  # Learn vocabulary and idf from training set.
    # list of -> (# of sentence, occurred words number    tf-idf-score)
    xtrain_tfv = tfv.transform(xtrain['text'])  # create sparse matrix with tf-idf probs
    xvalid_tfv = tfv.transform(xvalid['text'])
    return xtrain_tfv, xvalid_tfv


def get_all_features_df(xtrain, xvalid):
    xtrain_tfidf_wrd, xvalid_tfidf_wrd = get_tfidf_word_features(xtrain, xvalid)
    xtrain_tfidf_chr, xvalid_tfidf_chr = get_tfidf_char_features(xtrain, xvalid)
    xtrain_cnt, xvalid_cnt = get_count_features(xtrain, xvalid)

    xtrain_df = pd.concat([pd.DataFrame(xtrain_tfidf_wrd.toarray()), pd.DataFrame(xtrain_tfidf_chr.toarray()),
                           pd.DataFrame(xtrain_cnt.toarray())], axis=1)
    del xtrain_tfidf_wrd, xtrain_tfidf_chr, xtrain_cnt

    xvalid_df = pd.concat([pd.DataFrame(xvalid_tfidf_wrd.toarray()), pd.DataFrame(xvalid_tfidf_chr.toarray()),
                           pd.DataFrame(xvalid_cnt.toarray())], axis=1)
    del xvalid_tfidf_wrd, xvalid_tfidf_chr, xvalid_cnt

    return xtrain_df, xvalid_df
