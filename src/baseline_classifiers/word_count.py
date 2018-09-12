import sys

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.data_analysis.statistics import load_50_auth_data
from src.features.writing_style_features import preprocess_text
from src.utils.confusion import *
from src.utils.input_reader import *


# baseline_classifiers classifier
# Algorithm: Naeive Bayes on bag of words
# Features: word-count
def get_wc_feature(xtrain, xtest, ngram=1):
    ctv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, ngram))

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


# external data set should contain columns:
#  'author'
# 'author_label'= author labales serialized
# 'text'
if __name__ == '__main__':
    print("baseline_classifiers classifier")
    print("Algorithm: Naeive Bayes on bag of words")

    # LOAD DATA
    # train_df = load_50_auth_data()
    train_df = load_50_authors_preprocessed_data()
    referance_col = 'text'
    plots = False
    # plots = True
    ngram = 3
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
        if "ngram" in (arg_dict.keys()):
            ngram = int(arg_dict.get('ngram')[0])
        if "plots" in (arg_dict.keys()):
            plots = arg_dict.get('plots')[0]

    xtrain, xtest, ytrain, ytest = train_vali_split(train_df)
    xtrain = pd.DataFrame(xtrain[referance_col])
    xtrain = xtrain.rename(columns={referance_col: "text"})

    xtest = pd.DataFrame(xtest[referance_col])
    xtest = xtest.rename(columns={referance_col: "text"})

    # FEATURE CALCULATION- NB
    xtrain_ctv, xvalid_ctv = get_wc_feature(xtrain.text.values, xtest.text.values, ngram)

    clfnb = MultinomialNB()
    clfnb.fit(xtrain_ctv, ytrain)
    predictions_classes_nb = clfnb.predict(xvalid_ctv)
    predictions_nb = clfnb.predict_proba(xvalid_ctv)

    print("MultinomialNB measures:")
    print("logloss: %0.3f " % metrics.log_loss(ytest, predictions_nb))
    print("accuracy: %0.3f" % (np.sum(predictions_classes_nb == ytest) / len(ytest)))

    if plots == 'True':
        print("saving plots and outputs")
        # Compute and plot confusion matrix
        labels = list(set(train_df.author))
        cnf_matrix = confusion_matrix(ytest, predictions_classes_nb)
        np.set_printoptions(precision=2)
        fig = plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=labels,
                              title='Confusion matrix', normalize=True)
        fig.tight_layout()
        fig.savefig('Confusion.pdf', format='pdf')
        plt.close()
        print("confusion matrix saved to Confusion.pdf")

        # export predictions
        xtestdf = pd.DataFrame(data=xtest).reset_index(drop=True)
        ytestdf = pd.DataFrame(data=ytest).reset_index(drop=True)
        predictions_classes_nbdf = pd.DataFrame(data=predictions_classes_nb).reset_index(drop=True)
        df = pd.concat([xtestdf, predictions_classes_nbdf, ytestdf], axis=1)
        df = df.rename(columns={0: 'predictions_nb', 'author_label': 'ytest'})

        pairs = pd.DataFrame(train_df, columns=['author', 'author_label'])
        pairs = pairs.drop_duplicates()
        df_out = pd.merge(df, pairs, how='inner', left_on='predictions_nb', right_on='author_label')
        df_out = df_out.rename(index=str, columns={"author": "predictions_author"})
        df_out = pd.merge(df_out, pairs, how='inner', left_on='ytest', right_on='author_label')
        df_out = df_out.rename(index=str, columns={"author": "true_author"})
        df_out = df_out.drop(['author_label_x', 'author_label_y'], axis=1)

        pairs.to_csv('baseline_labels_encoding.ts', sep='\t', index=False)
        df_out.to_csv('baseline_evaluation_classes.tsv', sep='\t', index=False)

        df_errors = df_out[df_out.ytest != df_out.predictions_nb]
        df_errors.to_csv('baseline_evaluation_classes_errors.tsv', sep='\t')
        print(
            "baseline_labels_encoding.csv, baseline_evaluation_classes.tsv and baseline_evaluation_classes_errors.tsv saved")
