from keras.callbacks import EarlyStopping
from keras.layers import Dense, GlobalAveragePooling1D, Embedding
from keras.models import Sequential
from sklearn import metrics

from src.evaluations.evaluations import *
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import GlobalAveragePooling1D,Lambda,Input,GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D,TimeDistributed

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

    def create_model(self, input_dim, embedding_dims=32, optimizer='adam'):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims,input_length=256))
        # self.model.add(Dropout(0.3))
        # self.model.add(GlobalAveragePooling1D())
        # self.model.add(Dense(3, activation='softmax'))

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
        self.model.add(Dense(3, activation='softmax'))


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
                                   epochs=16, verbose=2,
                                   callbacks=[EarlyStopping(patience=4, monitor='val_loss')])

        predictions = self.model.predict_proba(docsvalid)
        predictions_classes = self.model.predict_classes(docsvalid)
        print("fasttext logloss: %0.3f " % metrics.log_loss(yvalid, predictions))
        print("fasttext business friendly output: %0.3f" % (
                np.sum(predictions_classes == yvalid) / len(yvalid)))
        return

    def predict(self, docstest):
        predictions = self.model.predict_proba(docstest)
        predictions_classes = self.model.predict_classes(docstest)
        return predictions, predictions_classes
        # TODO
