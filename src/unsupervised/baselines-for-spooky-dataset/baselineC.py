import gensim
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


def tag_docs(docs):
    tagged = docs.apply(lambda r: TaggedDocument(words=simple_preprocess(r['text']), tags=[r.id]), axis=1)
    return tagged


def train_doc2vec_model(tagged_docs, size, epochs, window=1):
    sents = tagged_docs.values
    doc2vec_model = Doc2Vec(sents, vector_size=size, window=window, epochs=epochs, dm=1)
    return doc2vec_model


def vec_for_learning(doc2vec_model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


if __name__ == '__main__':
    print("Baseline C unsupervised algorithm")
    print("Features: Doc2vec (new pipeline)")
    path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)

    # Model parameters
    feature_size = 150
    epochs_number = 22
    number_of_clusters = 3
    model_data_name = 'doc2vec_fsize[' + str(feature_size) + ']_clean[' + 'False' + ']_epoch[' + str(
        epochs_number) + '].model'

    # build tagged documents
    train_tagged = tag_docs(train_df)
    y = train_df.author_label

    # Train or Load model if exist
    if not os.path.isfile(model_data_name):
        model = train_doc2vec_model(train_tagged, feature_size, epochs_number)
        # saving the created model
        model.save(model_data_name)
        print('model saved')

    d2v_model = gensim.models.doc2vec.Doc2Vec.load(model_data_name)
    y_train, X = vec_for_learning(d2v_model, train_tagged)

    # Build plot of the data in 2D using PCA (DIM reduction):
    from sklearn.decomposition import PCA as sklearnPCA
    pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    x_transformed_2D = pd.DataFrame(pca.fit_transform(X))
    for i in range(0, number_of_clusters):
        plt.scatter(x_transformed_2D[train_df.author_label == i][0], x_transformed_2D[train_df.author_label == i][1],
                    c=np.random.rand(3, ),
                    label='author ' + str(i))
    plt.title('Sentence plot with DIM reduction (200D -> 2D)')
    plt.xlabel('X axis label')
    plt.ylabel('Y axis label')
    plt.legend()

    # Apply K-MEANS algorithm with 3 clusters: plot data with centroids
    kmeans_model = KMeans(number_of_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = pd.DataFrame(kmeans_model.fit_predict(X))
    centroids = kmeans_model.cluster_centers_
    centroidpoint = pca.transform(centroids)
    plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
    plt.show()  # need to manually close window

    # Run permutator to get best permutation
    perm = Permutator(number_of_clusters)
    scores = list()
    while perm.has_more_permurations():
        permuted_clusters_labels = cluster_labels[0].map(perm.get_permuration_series())
        permutation_score = metrics.accuracy_score(permuted_clusters_labels, pd.Series(train_df.author_label))
        scores.append(permutation_score)
        perm.set_next_permutation()

    print("Best score for clustering (unsupervised) " + str(max(scores)))
