from sklearn.decomposition import TruncatedSVD
import pandas as pd
from src.features import tf_idf_features

#used upon tf_idf x
def get_svd_word_features(trainx,testx):
    train_tfidf,test_tfidf=tf_idf_features.get_tfidf_word_features(trainx,testx)


    n_comp = 400
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(train_tfidf+test_tfidf)
    train_svd_wrd = pd.DataFrame(svd_obj.transform(train_tfidf))
    test_svd_wrd = pd.DataFrame(svd_obj.transform(test_tfidf))

    train_svd_wrd.columns = ['svd_word_' + str(i) for i in range(n_comp)]
    test_svd_wrd.columns = ['svd_word_' + str(i) for i in range(n_comp)]
    return train_svd_wrd,test_svd_wrd

    # train_df = pd.concat([train_df, train_svd_wrd], axis=1)
    # test_df = pd.concat([test_df, test_svd_wrd], axis=1)
    # del full_tfidf, train_tfidf, test_tfidf, train_svd_wrd, test_svd_wrd

def get_svd_char_features(trainx,testx):
    train_tfidf,test_tfidf=tf_idf_features.get_tfidf_char_features(trainx,testx)

    n_comp = 400
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(train_tfidf+test_tfidf)
    train_svd_chr = pd.DataFrame(svd_obj.transform(train_tfidf))
    test_svd_chr = pd.DataFrame(svd_obj.transform(test_tfidf))

    train_svd_chr.columns = ['svd_char_' + str(i) for i in range(n_comp)]
    test_svd_chr.columns = ['svd_char_' + str(i) for i in range(n_comp)]
    return train_svd_chr,test_svd_chr