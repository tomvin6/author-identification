from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers.core import Dropout
from sklearn import metrics
from src.features import fasttext_features
from src.features.fasttext_features import *
from src.features.writing_style_features import preprocess_text
from src.utils.input_reader import command_line_args, load_50_authors_preprocessed_data
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # train_df = load_50_auth_data()
    train_df = load_50_authors_preprocessed_data()
    referance_col = 'text_pos_tag_pairs'
    plots = False
    ngram = 1
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
        if "ngram" in (arg_dict.keys()):
            ngram = arg_dict.get('ngram')

    docs, tokenizer = create_docs(train_df[referance_col],referance_col=referance_col,n_gram_max=ngram)
    xtrain, xtest, ytrain, ytest = train_test_split(docs, train_df.author_label, stratify=train_df.author_label,
                                                    random_state=42,
                                                    test_size=0.3, shuffle=True)


    fsx,tokenizer = fasttext_features.obtain_fasttext_model(xtrain, ytrain, xtest, ytest, referance_col=referance_col,
                                                  create_doc=False)
    predictions, predictions_classes = fsx.predict(ytest)

    print("logloss: %0.3f " % metrics.log_loss(ytest, predictions))
    print("accuracy: %0.3f" % (metrics.accuracy_score(predictions_classes, ytest)))


class fasttext_classifier(object):
    def __init__(self):
        self.train_df = None
        self.train_X = None
        self.train_Y = None
        self.vslid_X = None
        self.valid_Y = None
        self.model = None
        self.hist = None
        self.tokenizer = None
        self.rare_train_words = []

    def create_model(self, input_dim, embedding_dims=32, optimizer='adam', classes=50):
        self.model = Sequential()
        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions

        self.model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims, input_length=256))

        self.model.add(Dropout(0.3))
        self.model.add(Conv1D(64,
                              5,
                              padding='valid',
                              activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(MaxPooling1D())
        self.model.add(Flatten())
        self.model.add(Dense(800, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(classes, activation='softmax'))

        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

        return

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def train(self, docstrain, ytrain, docsvalid, yvalid):
        self.train_X = docstrain
        self.train_Y = ytrain
        self.hist = self.model.fit(docstrain, ytrain,
                                   batch_size=16,
                                   validation_data=(docsvalid, yvalid),
                                   epochs=30, verbose=2,
                                   callbacks=[EarlyStopping(patience=4, monitor='val_loss')])

        predictions = self.model.predict_proba(docsvalid)
        predictions_classes = self.model.predict_classes(docsvalid)
        print("fasttext logloss: %0.3f " % metrics.log_loss(yvalid, predictions))
        print("accuracy: %0.3f" % (
                np.sum(predictions_classes == yvalid) / len(yvalid)))
        return

    def predict(self, docstest):
        predictions = self.model.predict_proba(docstest)
        predictions_classes = self.model.predict_classes(docstest)
        return predictions, predictions_classes

    def plot_train_vs_val(self):
        hist=self.model.history
        hist_dict=hist.history
        #plot loss
        fig=plt.figure()
        plt.subplot(211)
        val_loss=hist_dict.get('val_loss')
        val_loss_line=plt.plot(val_loss,label='val_loss')
        plt.legend()
        loss=hist_dict.get('loss')
        plt.plot(loss,label='train_loss')
        plt.legend()
        plt.title("train and validation loss")
        plt.ylabel("loss")

        #plot accuracy
        plt.subplot(212)
        val_acc=hist_dict.get('val_acc')
        plt.plot(val_acc, label='val_acc')
        plt.legend()
        acc=hist_dict.get('acc')
        plt.plot(acc, label='train_acc')
        plt.legend()
        plt.title("train and validation accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("step")

        fig.savefig("fast-text-itr-performance.pdf", format='pdf')




