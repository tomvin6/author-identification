from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA as sklearnPCA
from src.utils.input_reader import *
from src.utils.permutator import Permutator
from src.selectors.doc2vec_selector import *

# Model parameters
feature_size = 150
epochs_number = 22
number_of_clusters = 50
model_data_name = 'doc2vec_fsize[' + str(feature_size) + ']_clean[' + 'False' + ']_epoch[' + str(
    epochs_number) + '].model'

if __name__ == '__main__':
    print("Pipeline: unsupervised algorithm")
    print("50 readers input")
    print("Features: d2v, ")

    df = load_50_authors_data_sets_to_dict()
    # each transformer get input of above step
    # all except last step should be transformers

    # run doc2vec, transform for 2Dim vector for each document
    doc2vec_pipeline = Pipeline([
                         ("d2v", Doc2VecSelector(model_data_name, feature_size, epochs_number)),
                        ("pca", sklearnPCA(n_components=2))])

    # build vector of combined features
    # additional features should be added here
    combined_features = FeatureUnion([("d2v", doc2vec_pipeline)])

    all_features = combined_features.fit_transform(df)

    # run Kmeans on combined features
    kmeans_model = KMeans(number_of_clusters, init='k-means++',
           max_iter=300, n_init=10, random_state=0)
    cluster_labels = pd.DataFrame(kmeans_model.fit_predict(all_features))


    # Run permutator to get best permutation
    perm = Permutator(number_of_clusters)
    scores = list()
    while perm.has_more_permurations():
        permuted_clusters_labels = cluster_labels[0].map(perm.get_permuration_series())
        permutation_score = metrics.accuracy_score(permuted_clusters_labels, pd.Series(df.labels))
        scores.append(permutation_score)
        perm.set_next_permutation()

    print("Best score for clustering (unsupervised) " + str(max(scores)))
