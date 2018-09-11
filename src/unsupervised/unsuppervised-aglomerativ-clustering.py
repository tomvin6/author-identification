from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA as sklearnPCA, TruncatedSVD
from src.evaluations.evaluations import *
from src.selectors.average_words_selector import AverageWordsSelector
from src.selectors.item_selector import *
from src.utils.input_reader import *
from src.selectors.doc2vec_selector import *
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

# Model parameters
feature_size = 10  # d2v features
epochs_number = 22  # d2v features
number_of_clusters = 50  # clustering
model_data_name = 'doc2vec_fsize[' + str(feature_size) + ']_clean[' + 'False' + ']_epoch[' + str(
    epochs_number) + '].model'

def plot_dendrogram(model,X, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = hierarchy.linkage(X,agg.linkage)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


if __name__ == '__main__':
    print("Pipeline: unsupervised algorithm: with sentence transformation")
    print("50 readers input")
    print("aglomerative divisiv model(Diana)")
    print("Features: d2v, average word count, 3 4 5 grams")
    # original_data = load_50_authors_data_sentences_to_dict()
    features = load_50_authors_data_sets_to_dict()
    labels = features['author_label']
    df = features
    # run doc2vec, transform for 2Dim vector for each document
    # select each coordinate as a feature for clustering algorithm
    # doc2vec_pipeline = Pipeline([
    #                      ("d2v", Doc2VecSelector(model_data_name, feature_size, epochs_number))])

    tf_idf_3_grams = Pipeline([
                ('sel', ItemSelector(key='text')),
                ('tf', TfidfVectorizer(max_features=2000,
                          strip_accents='unicode', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')),
                ('svd', TruncatedSVD(n_components=25))
            ])
    tf_idf_4_grams = Pipeline([
                ('sel', ItemSelector(key='text_cleaned')),
                ('tf', TfidfVectorizer(max_features=1500,
                          strip_accents='unicode', token_pattern=r'\w{1,}',
                          ngram_range=(1, 4), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')),
                ('svd', TruncatedSVD(n_components=10))
            ])
    tf_idf_5_grams = Pipeline([
                ('sel', ItemSelector(key='text_pos_tag_pairs')),
                ('tf', TfidfVectorizer(max_features=1500,
                          strip_accents='unicode', token_pattern=r'\w{1,}',
                          ngram_range=(1, 5), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')),
                ('svd', TruncatedSVD(n_components=30))
            ])

    # average word count feature extraction pipeline
    word_count_pipeline = Pipeline([
                         ("word_count", AverageWordsSelector())])

    # build vector of combined features
    # additional features should be added to here
    combined_features = FeatureUnion([
            # ('word_count', word_count_pipeline),
            # ("d2vA", doc2vec_selectorA),
            #("d2v", doc2vec_pipeline),
            ("tfidf3", tf_idf_3_grams),
            #("tfidf4", tf_idf_4_grams),
            #("tfidf5", tf_idf_5_grams)
    ])

    print("Running pipelines to calculate model features \n")
    combined_features = combined_features.fit_transform(df)

    print("Running Agglomerative clustering on combined features \n")
    agg = AgglomerativeClustering(n_clusters =number_of_clusters, compute_full_tree =False,linkage="ward")

    cluster_labels = pd.DataFrame(agg.fit_predict(combined_features, y=labels))

    print_unsupervised_scores(labels, cluster_labels)

    # print("Accuracy: %0.3f" % (np.sum(cluster_labels == labels) / len(labels)))

    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(agg,combined_features, labels=agg.labels_)
    plt.show()


