import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.input_reader import load_data_sets, train_vali_split
from sklearn.cluster import KMeans
from sklearn import metrics
import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords


from src.utils.permutator import Permutator


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,
[self.labels_list[idx]])


# This function does all cleaning of data using two objects above
def nlp_clean(data):
    tokenizer = RegexpTokenizer(r'\w +')
    stopword_set = set(stopwords.words('english'))
    new_data = []
    for d in data:
        new_str = d.lower()
        dlist = tokenizer.tokenize(new_str)
        dlist = list(set(dlist).difference(stopword_set))
        new_data.append(dlist)
    return new_data


if __name__ == '__main__':
    print("Baseline unsupervised algorithm")
    print("Features: Doc2vec")
    path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)

    # Model parameters
    feature_size = 150
    epochs_number = 80
    clean = True
    model_data_name = 'doc2vec_fsize[' + str(feature_size) + ']_clean[' + str(clean) + ']_epoch[' + str(epochs_number) + '].model'

    # Pre process data
    if clean:
        data = nlp_clean(train_df.text.values)
    else:
        data = train_df.text.values

    # Convert sentences to vectors: doc2vec training
    # <<    THIS PHASE ONLY RUN ONCE, THEN GET MODEL FROM FILE     >>
    if not os.path.isfile(model_data_name):
        it = LabeledLineSentence(data, train_df.id)
        model = gensim.models.Doc2Vec(size=feature_size, min_count=1, alpha=0.025, min_alpha=0.025)
        model.build_vocab(it)

        for epochs_number in range(epochs_number):
            print('iteration' + str(epochs_number + 1))
            model.train(it, total_examples=model.corpus_count, epochs=model.iter)
            model.alpha -= 0.002
            model.min_alpha = model.alpha
        # saving the created model
        model.save(model_data_name)
        print('model saved')

    # loading the model from file to DataFrame
    d2v_model = gensim.models.doc2vec.Doc2Vec.load(model_data_name)
    X = pd.DataFrame(d2v_model.docvecs.doctag_syn0) # was vectors_docs

    # define number of clusters
    number_of_clusters = 3

    # Build plot of the data in 2D using PCA (DIM reduction):
    from sklearn.decomposition import PCA as sklearnPCA
    pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    x_transformed_2D = pd.DataFrame(pca.fit_transform(X))
    for i in range(0, number_of_clusters):
        plt.scatter(x_transformed_2D[train_df.author_label == i][0], x_transformed_2D[train_df.author_label == i][1], c=np.random.rand(3, ),
                    label='author ' + str(i))
    plt.title('Sentence plot with DIM reduction (200D -> 2D)')
    plt.xlabel('X axis label')
    plt.ylabel('Y axis label')
    plt.legend()

    # Apply K-MEANS algorithm with 3 clusters: plot data with centroids
    kmeans_model = KMeans(number_of_clusters, init='k-means++', max_iter= 300, n_init=10, random_state=0)
    cluster_labels = pd.DataFrame(kmeans_model.fit_predict(X))
    centroids = kmeans_model.cluster_centers_
    centroidpoint = pca.transform(centroids)
    plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
    plt.show()  # need to manually close window
    # cluster_labels.insert((cluster_labels.shape[1]), 'author', ytrain)

    perm = Permutator()
    scores = list()
    while perm.has_more_permurations():
        permuted_clusters_labels = cluster_labels[0].map(perm.get_permuration_series())
        permutation_score = metrics.accuracy_score(permuted_clusters_labels, pd.Series(train_df.author_label))
        scores.append(permutation_score)
        perm.set_next_permutation()

    print("Best score for clustering (unsupervised) " + str(max(scores)))

    # Apply Kmeans on transformed data (after PCA with 6 dimentions)
    pca2 = sklearnPCA(n_components=6)  # 2-dimensional PCA
    x_transformed_xD = pd.DataFrame(pca2.fit_transform(X))
    reducted_kmeans_model = KMeans(number_of_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    reducted_cluster_labels = pd.DataFrame(reducted_kmeans_model.fit_predict(x_transformed_xD))

    reducted_perm = Permutator()
    reducted_scores = list()
    while reducted_perm.has_more_permurations():
        permuted_clusters_labels = reducted_cluster_labels[0].map(reducted_perm.get_permuration_series())
        reducted_permutation_score = metrics.accuracy_score(permuted_clusters_labels, pd.Series(train_df.author_label))
        reducted_scores.append(reducted_permutation_score)
        reducted_perm.set_next_permutation()

    print("Best reducted score for clustering (unsupervised) " + str(max(reducted_scores)))
