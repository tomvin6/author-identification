from PIL import Image

from src.features.fasttext_features import *
from src.utils.input_reader import *

im = Image.open("grayscale.jpg")
print("advance classifier")
print("Algorithm: fast text")
print("None")


def convert_tabs_sep(path_prefix):
    replace_tabs_at_front(path_prefix + "train.txt", "beck-s", "beck-s###")
    replace_tabs_at_front(path_prefix + "train.txt", "farmer-d", "farmer-d###")
    replace_tabs_at_front(path_prefix + "train.txt", "kaminski-v", "kaminski-v###")
    replace_tabs_at_front(path_prefix + "test.txt", "beck-s", "beck-s###")
    replace_tabs_at_front(path_prefix + "test.txt", "farmer-d", "farmer-d###")
    replace_tabs_at_front(path_prefix + "test.txt", "kaminski-v", "kaminski-v###")
    replace_tabs_at_front(path_prefix + "validation.txt", "kaminski-v", "kaminski-v###")
    replace_tabs_at_front(path_prefix + "validation.txt", "farmer-d", "farmer-d###")
    replace_tabs_at_front(path_prefix + "validation.txt", "beck-s", "beck-s###")

# LOAD DATA
path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
X_train, xtest, y_train, ytest = train_vali_split(train_df)

data_train = X_train
data_test = xtest
# FEATURE CALCULATION
# bag of words
# creating a sparse matrix as BOW
# TODO: consider removing token pattern for better accuracy
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{0,}',
            ngram_range=(1, 2), lowercase=False, stop_words=None)

# build the pipeline
ppl = Pipeline([
              ('ngram', ctv),
              ('clf',   LogisticRegression(C=1.0))
      ])
# train the classifier
model = ppl.fit(data_train, y_train)
y_test = model.predict(data_test)
metrics.accuracy_score(y_test, ytest)

#X_train2 = ctv.fit_transform(X_train)
#test2  = ctv.transform(xtest)

# ctv.fit(list(X_train) + list(X_test)) # calculate parameters: mean, dif


# train the classifier
#classifier = LinearSVC()
#model = classifier.fit(X_train, y_train)

# test the classifier
#predictions_classes = model.predict(X_test)

from sklearn.pipeline import Pipeline, FeatureUnion
from src.features.writing_style_features import AdjExtractor, get_writing_style_features


pipeline = Pipeline([
    ('feats', FeatureUnion([
        ('ngram', ctv), # can pass in either a pipeline
        ('ave', AdjExtractor()) # or a transformer
    ])),
    ('clf', LogisticRegression(C=1.0))  # classifier
])

from sklearn.grid_search import GridSearchCV
pg = {'clf__C': [0.1, 1]}
#xx = get_writing_style_features(train_df)
grid = GridSearchCV(pipeline, param_grid=pg, cv=5)
grid.fit(data_train, y_train)
predictions_classes = grid.predict(xtest)
predictions_probs = grid.predict_proba(xtest)
metrics.accuracy_score(predictions_classes, ytest)
print("business friendly accuracy: %0.3f" % (np.sum(predictions_classes == ytest) / len(ytest)))

print ("logloss: %0.3f " % multiclass_logloss(ytest, predictions_probs))


