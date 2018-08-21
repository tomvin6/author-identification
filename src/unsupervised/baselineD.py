import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.utils import simple_preprocess

from src.utils.input_reader import load_data_sets
from sklearn.cluster import KMeans
from sklearn import metrics
from src.utils.permutator import Permutator


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


if __name__ == '__main__':
    print("Baseline D unsupervised algorithm")
    print("Features: TF-IDF")
    path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)

    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 4), stop_words='english', max_features=70)
    X = vectorizer.fit_transform(train_df['text'])

    # define number of clusters
    number_of_clusters = 3


    # Apply K-MEANS algorithm with 3 clusters: plot data with centroids
    kmeans_model = KMeans(number_of_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = pd.DataFrame(kmeans_model.fit_predict(X))
    centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]

    perm = Permutator()
    scores = list()
    while perm.has_more_permurations():
        permuted_clusters_labels = cluster_labels[0].map(perm.get_permuration_series())
        permutation_score = metrics.accuracy_score(permuted_clusters_labels, pd.Series(train_df.author_label))
        scores.append(permutation_score)
        perm.set_next_permutation()

    print("Best score for clustering (unsupervised) " + str(max(scores)))
