from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from src.data_analysis.statistics import load_50_auth_data
from src.utils.input_reader import *
from src.utils.confusion import *
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# baseline_classifiers classifier
# Algorithm: logistic regression
# Features: TF-IDF
def get_wc_feature(xtrain, xtest):
    ctv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 1))

    ctv.fit(list(xtrain) + list(xtest))
    xtrain_ctv = ctv.transform(xtrain)
    xvalid_ctv = ctv.transform(xtest)
    return xtrain_ctv, xvalid_ctv


def get_wc_featur_with_max_features(xtrain, xtest, max_features=200):
    ctv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), max_features=max_features)

    ctv.fit(list(xtrain) + list(xtest))
    xtrain_ctv = ctv.transform(xtrain)
    xvalid_ctv = ctv.transform(xtest)
    return xtrain_ctv, xvalid_ctv, ctv


if __name__ == '__main__':
    print("baseline_classifiers classifier")
    print("Algorithm: Log regression")
    print("Features: Word-count")

    # LOAD DATA
    # path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    # train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
    # xtrain, xtest, ytrain, ytest = train_vali_split(train_df)

    train_df = load_50_auth_data()
    xtrain, xtest, ytrain, ytest = train_vali_split(train_df)

    # FEATURE CALCULATION
    xtrain_ctv, xvalid_ctv = get_wc_feature(xtrain.text.values, xtest.text.values)

    # Fitting a simple Logistic Regression on Counts
    clflgr = LogisticRegression(C=1.0)
    clflgr.fit(xtrain_ctv, ytrain)
    predictions_classes_lgr = clflgr.predict(xvalid_ctv)
    predictions_lgr = clflgr.predict_proba(xvalid_ctv)

    print("LogisticRegression:")
    print("logloss: %0.3f " % metrics.log_loss(ytest, predictions_lgr))
    print("accuracy: %0.3f" % (np.sum(predictions_classes_lgr == ytest) / len(ytest)))

    # Compute and plot confusion matrix
    labels = list(set(train_df.author))
    cnf_matrix = confusion_matrix(ytest, predictions_classes_lgr)
    np.set_printoptions(precision=2)
    fig = plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels,
                          title='Confusion matrix')
    fig.savefig('Confusion.pdf', format='pdf')

    # export predictions
    d = {'text': xtest, 'predictions_lgr': predictions_classes_lgr, 'ytest': ytest}
    df = pd.DataFrame(data=d)
    df['predictions_lgr'] = df.predictions_lgr.apply(pd.to_numeric)
    df['ytest'] = df.ytest.apply(pd.to_numeric)

    pairs = pd.DataFrame(train_df, columns=['author', 'author_label'])
    pairs = pairs.drop_duplicates()
    df_out = pd.merge(df, pairs, how='inner', left_on='predictions_lgr', right_on='author_label')
    df_out = df_out.rename(index=str, columns={"author": "predictions_author"})
    df_out = pd.merge(df_out, pairs, how='inner', left_on='ytest', right_on='author_label')
    df_out = df_out.rename(index=str, columns={"author": "true_author"})
    df_out = df_out.drop(['author_label_x', 'author_label_y'], axis=1)

    pairs.to_csv('baseline_labels_encoding.csv', index=False)
    df_out.to_csv('baseline_evaluation_classes.tsv', sep='\t',index=False)

    # TODO- remove
    # clfnb = MultinomialNB()
    # clfnb.fit(xtrain_ctv, ytrain)
    # predictions_classes_nb = clfnb.predict(xvalid_ctv)
    # predictions_nb = clfnb.predict_proba(xvalid_ctv)
    #
    # print("MultinomialNB:" )
    # print("logloss: %0.3f " % metrics.log_loss(ytest, predictions_nb))
    # print("accuracy: %0.3f" % (np.sum(predictions_classes_nb == ytest) / len(ytest)))
