from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA as sklearnPCA, TruncatedSVD
from src.evaluations.evaluations import *
from src.selectors.average_words_selector import *
from src.selectors.item_selector import *
from src.utils.input_reader import *
from src.selectors.doc2vec_selector import *

# Model parameters
feature_size = 150  # d2v features
epochs_number = 22  # d2v features
number_of_clusters = 50  # clustering
model_data_name = 'doc2vec_fsize[' + str(feature_size) + ']_clean[' + 'False' + ']_epoch[' + str(
    epochs_number) + '].model'

if __name__ == '__main__':
    print("Pipeline: unsupervised algorithm baseline")
    print("50 readers input")
    print("Features: d2v, average word count, 3 4 5 grams")

    df = load_50_authors_data_sets_to_dict()
    labels = df['labels']

    # run doc2vec, transform for 2Dim vector for each document
    # select each coordinate as a feature for clustering algorithm

    meta_data = Pipeline([
                ('sel', ItemSelector(key='text')),
                ('meta', MetaStyleSelector())])


    # build vector of combined features
    # additional features should be added to here
    combined_features = FeatureUnion([
            ("tfidf3", meta_data)
    ])

    print("Running pipelines to calculate model features \n")
    combined_features = combined_features.fit_transform(df)

    print("Running K-means on combined features \n")
    km = KMeans(number_of_clusters, init='k-means++',
           max_iter=300, n_init=10, random_state=0)
    cluster_labels = pd.DataFrame(km.fit_predict(combined_features, y=labels))

    print_unsupervised_scores(labels, cluster_labels)


