from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA as sklearnPCA
from src.evaluations.logloss import *
from src.selectors.average_words_selector import AverageWordsSelector
from src.selectors.item_selector import ItemSelector
from src.utils.input_reader import *
from src.selectors.doc2vec_selector import *

# Model parameters
feature_size = 150  # d2v features
epochs_number = 22  # d2v features
number_of_clusters = 50  # clustering
model_data_name = 'doc2vec_fsize[' + str(feature_size) + ']_clean[' + 'False' + ']_epoch[' + str(
    epochs_number) + '].model'

if __name__ == '__main__':
    print("Pipeline: unsupervised algorithm")
    print("50 readers input")
    print("Features: d2v, average word count")

    df = load_50_authors_data_sets_to_dict()
    labels = df['labels']

    # run doc2vec, transform for 2Dim vector for each document
    # select each coordinate as a feature for clustering algorithm
    doc2vec_pipeline = Pipeline([
                         ("d2v", Doc2VecSelector(model_data_name, feature_size, epochs_number)),
                        ("pca", sklearnPCA(n_components=2))])
    doc2vec2Dim = doc2vec_pipeline.fit_transform(df)
    doc2vec_selectorA = Pipeline([
                        ("itemA", ItemSelector(doc2vec2Dim, 0))])
    doc2vec_selectorB = Pipeline([
                        ("itemB", ItemSelector(doc2vec2Dim, 1))])

    # average word count feature extraction pipeline
    word_count_pipeline = Pipeline([
                         ("word_count", AverageWordsSelector())])

    # build vector of combined features
    # additional features should be added to here
    combined_features = FeatureUnion([
            ('word_count', word_count_pipeline),
            ("d2vA", doc2vec_selectorA),
            ("d2vB", doc2vec_selectorB)
    ])

    print("Running pipelines to calculate model features \n")
    combined_features = combined_features.fit_transform(df)

    print("Running K-means on combined features \n")
    km = KMeans(number_of_clusters, init='k-means++',
           max_iter=300, n_init=10, random_state=0)
    cluster_labels = pd.DataFrame(km.fit_predict(combined_features, y=labels))

    print("scores:")
    print("Purity score: %0.3f" % purity_score(labels, cluster_labels))
    print("Normalize Mutual score: %0.3f" % normalized_mutual_score(labels, cluster_labels))
    print("\n")
