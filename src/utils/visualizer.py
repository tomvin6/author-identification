from yellowbrick.text import TSNEVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.input_reader import *


# LOAD DATA
path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
xtrain, xtest, ytrain, ytest = train_vali_split(train_df)

print('average sentence length: %s' % train_df['text'].map(lambda x: len(x.split())).mean())
print('min sentence length: %s' % train_df['text'].map(lambda x: len(x.split())).min())

# Load the data and create document vectors
tfv = TfidfVectorizer(min_df=3, max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1)

docs = tfv.fit_transform(list(xtrain) + list(xtest))
labels = list(ytrain) + list(ytest)

# Create the visualizer and draw the vectors
tsne = TSNEVisualizer(labels=['Author A', 'Author B', 'Author C'])
tsne.fit(docs, labels)
tsne.poof()