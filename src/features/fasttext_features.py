from keras.layers import Dense, GlobalAveragePooling1D, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn import model_selection

import src.baseline_classifiers.fasttext as fasttext
from src.baseline_classifiers.svm_tfidf import *
from src.evaluations.evaluations import *


# preproceeings are:
# Separate punctuation from words
# Remove lower frequency words ( <= 2)
# Cut a longer document which contains 256 words
def preprocess(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, ' {} '.format(sign))
    return text


def create_docs(data, n_gram_max=2, tokenizer=None, train_mode=True, referance_col='text'):
    df = pd.DataFrame(data=data, columns=[referance_col])
    rare_train_words = []

    # create N grams + separate punctuation from words
    def add_ngram(q, n_gram_max):
        ngrams = []
        for n in range(2, n_gram_max + 1):
            for w_index in range(len(q) - n + 1):
                ngrams.append('--'.join(q[w_index:w_index + n]))
        return q + ngrams

    docs = []
    for doc in df[referance_col]:
        doc = preprocess(doc).split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))

    min_count = 2
    if tokenizer is None:
        tokenizer = Tokenizer(lower=False, filters='')
        tokenizer.fit_on_texts(docs)

    if train_mode:
        # remove low frequency words
        num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])
        tokenizer = Tokenizer(num_words=num_words, lower=False, filters='')
        tokenizer.fit_on_texts(docs)

    docs = tokenizer.texts_to_sequences(docs)
    maxlen = 256
    # cat long sentences and
    docs = pad_sequences(sequences=docs, maxlen=maxlen)

    if train_mode:
        return docs, tokenizer
    else:
        return docs


# keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
# embedding_dims- change to 32?!
def create_model(input_dim, embedding_dims=20, optimizer='adam'):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation='softmax'))

    # optimizer- Adam or SGD?!
    # optimizer = SGD(lr=0.01, decay=0.0005, momentum=0.85, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


# get fsx featurew for trainig and validation set in a cross validation methodology
def get_fasttext_features(xtrain, ytrain, xvalid, yvalid, referance_col='text', lbl_prefix='fastext_'):
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([xtrain.shape[0], len(set(ytrain))])

    fsx = fasttext.fasttext_classifier()
    docstrain, tokenizer = create_docs(data=xtrain[referance_col], referance_col=referance_col)
    fsx.set_tokenizer(tokenizer)

    docstest = create_docs(data=xvalid[referance_col], tokenizer=fsx.tokenizer, train_mode=False,
                           referance_col=referance_col)
    input_dim = np.max(docstrain) + 1
    fsx.create_model(input_dim)

    # split training set to 5 folds
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    cv_cnt = 1
    for dev_index, val_index in kf.split(docstrain):
        print("CV fsx:" + str(cv_cnt))
        cv_cnt += 1

        dev_X, val_X = docstrain[dev_index], docstrain[val_index]
        # dev_y, val_y = ytrain.iloc[dev_index], ytrain.iloc[val_index]
        dev_y, val_y = ytrain.iloc[dev_index], ytrain.iloc[val_index]

        fsx.train(dev_X, dev_y, val_X, val_y)
        prob_val_y, cls_val_y = fsx.predict(val_X)
        prob_test_y, cls_test_y = fsx.predict(docstest)

        pred_full_test = pred_full_test + prob_test_y
        pred_train[val_index, :] = prob_val_y
        cv_scores.append(metrics.log_loss(val_y, prob_val_y))
    print("Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.

    columns = [lbl_prefix + str(i) for i in range(len(set(ytrain)))]
    return pd.DataFrame(columns=columns, data=pred_train), pd.DataFrame(columns=columns, data=pred_full_test)


# this methos to be used to save model created on training set, for new row currently not in DB
def obtain_fasttext_model(xtrain, ytrain, xvalid, yvalid, referance_col='text',create_doc=True):

    fsx = fasttext.fasttext_classifier()

    if create_doc:
        docstrain, tokenizer = create_docs(data=xtrain[referance_col], referance_col=referance_col)
        fsx.set_tokenizer(tokenizer)
        docstest = create_docs(data=xvalid[referance_col], tokenizer=fsx.tokenizer, train_mode=False,
                           referance_col=referance_col)
    else:
        docstrain=xtrain
        docstest=xvalid

    input_dim = np.max(docstrain) + 1
    fsx.create_model(input_dim,classes=len(set(ytrain)))

    fsx.train(docstrain, ytrain, docstest, yvalid)
    return fsx


# def get_fasttext_features1(docstrain, ytrain, docsvalid, yvalid, input_dim=0):
#     model = create_model(input_dim)
#     hist = model.fit(docstrain, ytrain,
#                      batch_size=16,
#                      validation_data=(docsvalid, yvalid),
#                      epochs=16,
#                      callbacks=[EarlyStopping(patience=4, monitor='val_loss')])
#     model.predict(docsvalid)
#     # CHANGED hist TO HIST.MODEL
#     predictions = hist.model.predict_proba(docsvalid)
#     predictions_classes = hist.model.predict_classes(docsvalid)
#
#
#     print("logloss: %0.3f " % metrics.log_loss(yvalid, predictions))
#     print("accuracy: %0.3f" % (metrics.accuracy_score(predictions_classes, yvalid)))
#
#     return predictions, predictions_classes
#
#
# def test_fasttest_cls1():
#     # read input data
#     path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
#     train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
#     docs = create_docs(train_df['text'])
#     xtrain, xvalid, ytrain, yvalid = train_test_split(docs, train_df.author_label, stratify=train_df.author_label,
#                                                       random_state=42,
#                                                       test_size=0.2, shuffle=True)
#     get_fasttext_features1(xtrain, ytrain, xvalid, yvalid, np.max(docs) + 1)

if __name__ == '__main__':
    path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
    # docs = create_docs(train_df['text'])
    xtrain, xvalid, ytrain, yvalid = train_test_split(train_df, train_df['author_label'],
                                                      stratify=train_df['author_label'],
                                                      random_state=42,
                                                      test_size=0.2, shuffle=True)
    get_fasttext_features(xtrain, ytrain, xvalid, yvalid, lbl_prefix='fastext_')
