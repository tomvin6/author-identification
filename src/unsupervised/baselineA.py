import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.baseline_classifiers.word_count import get_wc_featur_with_max_features
from src.utils.input_reader import load_data_sets, train_vali_split
from sklearn.cluster import KMeans
from sklearn import metrics

from src.utils.permutator import Permutator

if __name__ == '__main__':
    print("Baseline unsupervised algorithm")
    print("Features: Word-count (")
    path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
    xtrain, xtest, ytrain, ytest = train_vali_split(train_df)

    # get 200 features from count vectorizer
    xtrain_ctv, xvalid_ctv, ctv = get_wc_featur_with_max_features(xtrain, xtest)
    X = pd.DataFrame(xtrain_ctv.toarray(), columns=ctv.get_feature_names())
    X_array = X.iloc[:,:].values

    # define number of clusters
    number_of_clusters = 3


    # Plot data in 2D using PCA (DIM reduction):
    from sklearn.decomposition import PCA as sklearnPCA
    pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    x_transformed_2D = pd.DataFrame(pca.fit_transform(X_array))
    for i in range(0, number_of_clusters):
        plt.scatter(x_transformed_2D[ytrain == i][0], x_transformed_2D[ytrain == i][1], c=np.random.rand(3, ),
                    label='author ' + str(i))
    plt.title('Sentence plot with DIM reduction (200D -> 2D)')
    plt.xlabel('X axis label')
    plt.ylabel('Y axis label')
    plt.legend()
    plt.show()  # need to manually close window

    # Apply K-MEANS algorithm with 3 clusters
    kmeans_model = KMeans(number_of_clusters, init='k-means++', max_iter= 300, n_init=10, random_state=0)
    cluster_labels = pd.DataFrame(kmeans_model.fit_predict(X))
    centers = kmeans_model.cluster_centers_
    # cluster_labels.insert((cluster_labels.shape[1]), 'author', ytrain)

    perm = Permutator()
    scores = list()
    while perm.has_more_permurations():
        permuted_clusters_labels = cluster_labels[0].map(perm.get_permuration_series())
        permutation_score = metrics.accuracy_score(permuted_clusters_labels, pd.Series(ytrain))
        scores.append(permutation_score)
        perm.set_next_permutation()

    print("Best score for clustering (unsupervised) " + str(max(scores)))
