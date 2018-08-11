import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas

# import the dataset
# LOAD DATA
from src.baseline_classifiers.word_count import get_wc_featur_with_max_features
from src.utils.input_reader import load_data_sets, train_vali_split
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import random

if __name__ == '__main__':
    print("baseline unsupervised algo")
    print("Features: Word-count")
    path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
    xtrain, xtest, ytrain, ytest = train_vali_split(train_df)

    # FEATURE CALCULATION
    xtrain_ctv, xvalid_ctv, ctv = get_wc_featur_with_max_features(xtrain, xtest)
    X = pd.DataFrame(xtrain_ctv.toarray(), columns=ctv.get_feature_names())
    # using elbow method to find number of clusters ?
    X = X.iloc[:,:].values
    # apply KMEANS with num of clusters
    number_of_clusters = 3
    kmeans = KMeans(number_of_clusters, init='k-means++', max_iter= 300, n_init=10, random_state=0)
    ykmeans = kmeans.fit_predict(X)

    from sklearn.decomposition import PCA as sklearnPCA
    pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    transformed = pd.DataFrame(pca.fit_transform(X))

    # visualizing the clusters
    for i in range(0, number_of_clusters):
        plt.scatter(transformed[ykmeans == i][0], transformed[ykmeans == i][1], c=np.random.rand(3,), label='Cluster ' + str(i))
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label='centroids')
    plt.title('clusters plot')
    plt.xlabel('x axis label')
    plt.ylabel('y axis label')
    plt.legend()
    plt.show()
    print ("")
