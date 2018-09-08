from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA as sklearnPCA, TruncatedSVD
from src.evaluations.evaluations import *
from src.selectors.average_words_selector import AverageWordsSelector
from src.selectors.item_selector import *
from src.utils.input_reader import *
from src.selectors.doc2vec_selector import *

# Model parameters
feature_size = 10  # d2v features
epochs_number = 22  # d2v features
number_of_clusters = 50  # clustering


if __name__ == '__main__':
    print("Pipeline: unsupervised algorithm: with sentence transformation")
    print("50 readers input")
    print("Features: d2v, average word count, 3 4 5 grams")
    original_data = load_50_authors_data_sentences_to_dict()
    features = load_50_authors_preprocessed_data()
    labels = original_data['author_label']
    df = features

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

    # build vector of combined features
    # additional features should be added to here
    combined_features = FeatureUnion([
            ("tfidf3", tf_idf_3_grams),
            ("tfidf4", tf_idf_4_grams),
            ("tfidf5", tf_idf_5_grams)
    ])

    print("Running pipelines to calculate model features \n")
    combined_features = combined_features.fit_transform(df)

    print("Running K-means on combined features \n")
    km = KMeans(number_of_clusters, init='k-means++',
           max_iter=300, n_init=10, random_state=0)
    cluster_labels = pd.DataFrame(km.fit_predict(combined_features, y=labels))

    print_unsupervised_scores(labels, cluster_labels)

