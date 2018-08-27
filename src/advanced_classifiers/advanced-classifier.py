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

path_prefix = ".." + os.sep + ".." + os.sep + "additional-corpus-input" + os.sep
# convert_tabs_sep(path_prefix)
df_data, test_df, sample_df = load_txt_data_sets(path_prefix + "train.txt", path_prefix + "test.txt", None)
df = df_data
author_col = df.author_label

text_feature = df[1].to_frame(name="text")
author_numbers = df['author_label'].to_frame(name="author")
docs = create_docs(text_feature, train_mode=False)
xtrain, xvalid, ytrain, yvalid = train_test_split(docs, author_numbers, stratify=author_numbers, random_state=42, test_size=0.2)
X_test = xvalid
y_test = yvalid
# return a matrix of 3 classes and probability for each class:
predictions, predictions_classes = get_fasttext_features1(xtrain, ytrain, xvalid, yvalid, np.max(docs) + 1)


print ("accuracy: %0.3f " % metrics.accuracy_score(y_test, predictions_classes))
print ("logloss: %0.3f " % metrics.log_loss(y_test, predictions))

# fasttext2 = Pipeline([
#                 ('selector', TransperentSelector(1)),
#                  ('standard', StandardScaler())
#             ])
# fasttext2.fit_transform(X_train)
# fasttext3 = Pipeline([
#                 ('selector', TransperentSelector(2)),
#                  ('standard', StandardScaler())
#             ])
# fasttext3.fit_transform(X_train)
#
# chr_len = Pipeline([
#                 ('selector', NumberSelector(key='char_count')),
#                  ('standard', StandardScaler())
#             ])
# chr_len.fit_transform(X_train)
#
#
# chr_len = Pipeline([
#                 ('selector', NumberSelector(key='char_count')),
#                  ('standard', StandardScaler())
#             ])
# chr_len.fit_transform(X_train)
# word_len = Pipeline([
#                 ('selector', NumberSelector(key='word_count')),
#                  ('standard', StandardScaler())
#             ])
# word_len.fit_transform(X_train)
# unique_word_fraction = Pipeline([
#                 ('selector', NumberSelector(key='unique_word_fraction')),
#                  ('standard', StandardScaler())
#             ])
# unique_word_fraction.fit_transform(X_train)
# punctuations_fraction = Pipeline([
#                 ('selector', NumberSelector(key='punctuations_fraction')),
#                  ('standard', StandardScaler())
#             ])
# punctuations_fraction.fit_transform(X_train)
# fraction_noun = Pipeline([
#                 ('selector', NumberSelector(key='fraction_noun')),
#                  ('standard', StandardScaler())
#             ])
# fraction_noun.fit_transform(X_train)
# fraction_adj = Pipeline([
#                 ('selector', NumberSelector(key='fraction_adj')),
#                  ('standard', StandardScaler())
#             ])
# fraction_adj.fit_transform(X_train)
# fraction_verbs = Pipeline([
#                 ('selector', NumberSelector(key='fraction_verbs')),
#                  ('standard', StandardScaler())
#             ])
# fraction_verbs.fit_transform(X_train)
#
# ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{0,}',
#             ngram_range=(1, 2), lowercase=False, stop_words=None)
# text = Pipeline([
#                 ('selector', TextSelector(key='text')),
#                 ('tfidf', ctv)
#             ])
# text.fit_transform(X_train)
#
# from sklearn.pipeline import FeatureUnion
# raw_features = FeatureUnion([('unique_word_fraction', unique_word_fraction),
#                              ('punctuations_fraction', punctuations_fraction),
#                              ('char_count', chr_len),
#                              ('word_count', word_len),
#                              ('fraction_noun', fraction_noun),
#                              ('fraction_adj', fraction_adj),
#                              ('fraction_verbs', fraction_verbs),
#                              ('count_vector', text)])
#
# combined_raw_features = Pipeline([('raw_features', raw_features)])
# X_features = combined_raw_features.fit_transform(X_train)

# fasttext_features = Pipeline([
#                 ('selector', DummySelector())
#             ])
# transformed_features = fasttext_features.fit_transform(pd.DataFrame(predictions))
#
# all_features = FeatureUnion([('raw_features_pipeline', combined_raw_features),
#                              ('fasttext1', fasttext_features)
#                              ])


# build the pipeline
from sklearn.ensemble import RandomForestClassifier
# pipeline = Pipeline([
#     ('all_features', all_features),
#     ('classifier', LogisticRegression()),
# ])

# pipeline.fit(X_train, y_train)

# print ("accuracy " + metrics.accuracy_score(y_test, predictions))
#
# from sklearn.grid_search import GridSearchCV
# pg = {'clf__C': [0.1, 1]}
#
# grid = GridSearchCV(pipeline, param_grid=pg, cv=5)
# grid.fit(data_train, y_train)
# predictions_classes = grid.predict(xtest)
# predictions_probs = grid.predict_proba(xtest)
# metrics.accuracy_score(predictions_classes, ytest)
# print("business friendly accuracy: %0.3f" % (np.sum(predictions_classes == ytest) / len(ytest)))
#
# print ("logloss: %0.3f " % multiclass_logloss(ytest, predictions_probs))
#
#
