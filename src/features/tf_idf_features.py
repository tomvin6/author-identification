from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#get tf-idf features on a given trainning and validation sets
def get_tfidf_word_features(xtrain, xtest):
    tfv = TfidfVectorizer(min_df=3, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')
    # Fitting TF-IDF to both training and test sets (semi-supervised learning)
    tfv.fit_transform(list(xtrain) + list(xtest))  # Learn vocabulary and idf from training set.
    # list of -> (# of sentence, occurred words number    tf-idf-score)
    xtrain_tfv = tfv.transform(xtrain)  # create sparse matrix with tf-idf probs
    xvalid_tfv = tfv.transform(xtest)
    return xtrain_tfv, xvalid_tfv

def get_tfidf_char_features(xtrain, xtest):
    tfv = TfidfVectorizer(min_df=3, max_features=None,
                          strip_accents='unicode', analyzer='char', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')
    # Fitting TF-IDF to both training and test sets (semi-supervised learning)
    tfv.fit_transform(list(xtrain) + list(xtest))  # Learn vocabulary and idf from training set.
    # list of -> (# of sentence, occurred words number    tf-idf-score)
    xtrain_tfv = tfv.transform(xtrain)  # create sparse matrix with tf-idf probs
    xvalid_tfv = tfv.transform(xtest)
    return xtrain_tfv, xvalid_tfv

def get_count_features(xtrain, xtest):
    tfv = CountVectorizer(stop_words='english', ngram_range=(1,3))
    # Fitting TF-IDF to both training and test sets (semi-supervised learning)
    tfv.fit_transform(list(xtrain) + list(xtest))  # Learn vocabulary and idf from training set.
    # list of -> (# of sentence, occurred words number    tf-idf-score)
    xtrain_tfv = tfv.transform(xtrain)  # create sparse matrix with tf-idf probs
    xvalid_tfv = tfv.transform(xtest)
    return xtrain_tfv, xvalid_tfv