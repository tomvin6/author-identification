from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA as sklearnPCA, TruncatedSVD
from src.evaluations.evaluations import *
from src.selectors.average_words_selector import AverageWordsSelector, MetaStyleSelector
from src.selectors.item_selector import *
from src.utils.input_reader import *
import matplotlib.pyplot as plt

# Model default parameters
default_number_of_clusters = 50  # clustering
default_dim_reduction_for_word_ngram = 20
default_draw_clustering_output = True
features = 'text_cleaned'


if __name__ == '__main__':
    print("Pipeline: Best score unsupervised algorithm")
    print("Format: Original Messages")
    print("Input: 50 readers input")
    print("Features: 3=grams, 4-grams, 5-grams, cleaned text, tf-idf, pca, characters + word data")

    number_of_clusters, word_dim_reduction, is_draw = get_main_parameters(sys.argv, default_number_of_clusters,
                                                                          default_dim_reduction_for_word_ngram,
                                                                          default_draw_clustering_output)
    df = load_50_authors_data_sets_to_dict()
    labels = df['author_label']

    # extract features from data set
    tf_idf_3_grams = Pipeline([
                ('extract', MetaStyleSelector("style_features_full_text_test_set.pkl")),
                ('sel', ItemSelector(key=features)),
                ('tf', TfidfVectorizer(max_features=1000,
                                       strip_accents='unicode', token_pattern=r'\w{1,}',
                                       ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                                       stop_words='english')),
                ('svd', TruncatedSVD(n_components=word_dim_reduction))
    ])

    tf_idf_4_grams = Pipeline([
                ('extract', MetaStyleSelector("style_features_full_text_test_set.pkl")),
                ('sel', ItemSelector(key=features)),
                ('tf', TfidfVectorizer(max_features=950,
                          strip_accents='unicode', token_pattern=r'\w{1,}',
                          ngram_range=(1, 4), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')),
                ('svd', TruncatedSVD(n_components=word_dim_reduction))
            ])
    tf_idf_5_grams = Pipeline([
                ('sel', ItemSelector(key='text')),
                ('tf', TfidfVectorizer(max_features=1500,
                          strip_accents='unicode', token_pattern=r'\w{1,}',
                          ngram_range=(1, 5), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')),
                ('svd', TruncatedSVD(n_components=word_dim_reduction))
            ])

    tf_idf_letters_grams = Pipeline([
                ('sel', ItemSelector(key='text')),
                ('tf', TfidfVectorizer(max_features=100,
                          strip_accents='unicode', analyzer='char',
                          ngram_range=(1, 3))),
                ('svd', TruncatedSVD(n_components=5))
            ])

    # average word count feature extraction pipeline
    word_count_pipeline = Pipeline([
                         ("word_count", AverageWordsSelector())])

    # build vector of combined features
    # additional features should be added to here
    combined_features = FeatureUnion([
            ("tfidf3letter", tf_idf_letters_grams),
            ("tfidf3word", tf_idf_3_grams),
            ("tfidf4word", tf_idf_4_grams),
            ("tfidf5word", tf_idf_5_grams)
    ])

    print("Running pipelines to calculate model features \n")
    combined_features = combined_features.fit_transform(df)

    print("Running K-means on combined features \n")
    km = KMeans(number_of_clusters, init='k-means++',
           max_iter=350, n_init=30, random_state=0)
    cluster_labels = pd.DataFrame(km.fit_predict(combined_features, y=labels))

    print_unsupervised_scores(labels, cluster_labels)

    if is_draw:
        # Plot data in 2D using PCA (DIM reduction):
        from sklearn.decomposition import PCA as sklearnPCA
        pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
        x_transformed_2D = pd.DataFrame(pca.fit_transform(combined_features))
        for i in range(0, number_of_clusters):
            plt.scatter(x_transformed_2D[labels == i][0], x_transformed_2D[labels == i][1], c=np.random.rand(3, ),
                        label='author ' + str(i))
        plt.title('Authors clustering with DIM reduction (2D)')
        plt.xlabel('X axis label')
        plt.ylabel('Y axis label')
        plt.legend()
        plt.show()  # need to manually close window
