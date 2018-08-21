from sklearn.decomposition import TruncatedSVD
import pandas as pd
from src.features import tf_idf_features
from scipy.sparse import vstack


# used upon tf_idf x
def get_svd_word_features(trainx, testx):
    train_tfidf, test_tfidf = tf_idf_features.get_tfidf_word_features(trainx, testx)
    return get_svd_features(train_tfidf, test_tfidf, 'svd_word_')

    # train_df = pd.concat([train_df, train_svd_wrd], axis=1)
    # test_df = pd.concat([test_df, test_svd_wrd], axis=1)
    # del full_tfidf, train_tfidf, test_tfidf, train_svd_wrd, test_svd_wrd


def get_svd_char_features(trainx, testx):
    train_tfidf, test_tfidf = tf_idf_features.get_tfidf_char_features(trainx, testx)
    return get_svd_features(train_tfidf, test_tfidf, 'svd_char_')


def get_svd_features(trainx, testx, lbl_prefix):
    x_train_test = vstack((trainx, testx), format='csr')
    n_comp = 200
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(x_train_test)
    train_svd_chr = pd.DataFrame(svd_obj.transform(trainx))
    test_svd_chr = pd.DataFrame(svd_obj.transform(testx))

    train_svd_chr.columns = [lbl_prefix + str(i) for i in range(n_comp)]
    test_svd_chr.columns = [lbl_prefix + str(i) for i in range(n_comp)]
    return train_svd_chr, test_svd_chr
