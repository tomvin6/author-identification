from nltk import word_tokenize
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, preprocessing, decomposition, model_selection, metrics, pipeline
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from src.features.fasttext_features import get_fasttext_features1, get_fasttext_features, create_docs, test_fasttest_cls1
from src.features.writing_style_features import get_writing_style_features
import pandas as pd
import os

from src.evaluations.evaluations import multiclass_logloss
from src.utils.input_reader import train_vali_split, load_data_sets
from src.advanced_classifiers.columnSelectors import *

# baseline_classifiers classifier
# Algorithm: logistic regression
# Features: TF-IDF

print("advance classifier")
print("Algorithm: Log regression")
print("Pre processing: removal of english stop words turned off, lowercase turn off")
print("Features: Word-count")



# LOAD DATA
path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
df_data, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)

cnt_srs = train_df['author'].value_counts()

df = df_data #df_data.head(n = 20)
#df = df_data
author_col = df['author_label']
df = get_writing_style_features(df)
df['author'] = author_col
features = [c for c in df.columns.values if c  not in ['id','author', 'author_label']]
X_train, X_test, y_train, y_test = train_test_split(df[features], df['author'], test_size=0.2, random_state=42)


text_feature = X_train['text']
author_numbers = pd.DataFrame(y_train)

# xtrain, xvalid, ytrain, yvalid = train_test_split(docs, author_numbers, stratify=author_numbers, random_state=42, test_size=0.2)

# return a matrix of 3 classes and probability for each class:
# predictions, predictions_classes = get_fasttext_features1(xtrain, ytrain, xvalid, yvalid, np.max(docs) + 1)

def transform_raw_features(X_train):
    chr_len = Pipeline([
                    ('selector', NumberSelector(key='char_count')),
                     ('standard', StandardScaler())
                ])
    chr_len.fit_transform(X_train)

    chr_len = Pipeline([
                    ('selector', NumberSelector(key='char_count')),
                     ('standard', StandardScaler())
                ])
    chr_len.fit_transform(X_train)
    word_len = Pipeline([
                    ('selector', NumberSelector(key='word_count')),
                     ('standard', StandardScaler())
                ])
    word_len.fit_transform(X_train)
    unique_word_fraction = Pipeline([
                    ('selector', NumberSelector(key='unique_word_fraction')),
                     ('standard', StandardScaler())
                ])
    unique_word_fraction.fit_transform(X_train)
    punctuations_fraction = Pipeline([
                    ('selector', NumberSelector(key='punctuations_fraction')),
                     ('standard', StandardScaler())
                ])
    punctuations_fraction.fit_transform(X_train)
    fraction_noun = Pipeline([
                    ('selector', NumberSelector(key='fraction_noun')),
                     ('standard', StandardScaler())
                ])
    fraction_noun.fit_transform(X_train)
    fraction_adj = Pipeline([
                    ('selector', NumberSelector(key='fraction_adj')),
                     ('standard', StandardScaler())
                ])
    fraction_adj.fit_transform(X_train)
    fraction_verbs = Pipeline([
                    ('selector', NumberSelector(key='fraction_verbs')),
                     ('standard', StandardScaler())
                ])
    fraction_verbs.fit_transform(X_train)

    ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{0,}',
                ngram_range=(1, 2), lowercase=False, stop_words=None)
    text = Pipeline([
                    ('selector', TextSelector(key='text')),
                    ('tfidf', ctv)
                ])
    text.fit_transform(X_train)

    # transform all data with new features
    from sklearn.pipeline import FeatureUnion
    raw_features = FeatureUnion([('unique_word_fraction', unique_word_fraction),
                                 ('punctuations_fraction', punctuations_fraction),
                                 ('char_count', chr_len),
                                 ('word_count', word_len),
                                 ('fraction_noun', fraction_noun),
                                 ('fraction_adj', fraction_adj),
                                 ('fraction_verbs', fraction_verbs),
                                 ('count_vector', text)])

    combined_raw_features = Pipeline([('raw_features', raw_features)])
    X_features = combined_raw_features.fit_transform(X_train)
    return combined_raw_features


combined_raw_features_for_train = transform_raw_features(X_train)


pipeline = Pipeline([
    ('all_features', combined_raw_features_for_train),
    ('classifier', LogisticRegression())
])
pipeline.fit(X_train, y_train)
raw_feature_output = pipeline.predict_proba(X_train)

docs_train = create_docs(text_feature, train_mode=False)
fasttext_model = get_fast_text_model(docs_train, y_train)

fasttext = Pipeline([
    ('ft_docs', FastTextPredictor(fasttext_model)),
])
fast_output = fasttext.fit_transform(docs_train, y_train)
result = pd.concat([fast_output, pd.DataFrame(raw_feature_output)], axis=1, sort=False)

clf = LogisticRegression(C=1.0)
clf.fit(result, y_train)

# calculate results on validation set
docs_test = create_docs(X_test['text'], train_mode=False)
raw_predictions = pipeline.predict_proba(X_test)
fasttext_predictions = fasttext.fit_transform(docs_test, y_train)
test_valid_features = pd.concat([fasttext_predictions, pd.DataFrame(raw_predictions)], axis=1, sort=False)

final_predictions_classes = clf.predict(test_valid_features)
final_predictions_probs = clf.predict_proba(test_valid_features)

print("accuracy: " + str(metrics.accuracy_score(final_predictions_classes, y_test)))
print("log-loss: " + str(metrics.log_loss(y_test, final_predictions_probs)))

