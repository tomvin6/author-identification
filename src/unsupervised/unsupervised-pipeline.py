from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA as sklearnPCA, TruncatedSVD
from src.evaluations.evaluations import *
from src.selectors.average_words_selector import AverageWordsSelector, MetaStyleSelector
from src.selectors.item_selector import *
from src.utils.input_reader import *
import matplotlib.pyplot as plt

# Model parameters
feature_size = 150  # d2v features
epochs_number = 22  # d2v features
number_of_clusters = 50  # clustering
model_data_name = 'doc2vec_fsize[' + str(feature_size) + ']_clean[' + 'False' + ']_epoch[' + str(
    epochs_number) + '].model'
features = 'text_cleaned'


if __name__ == '__main__':
    print("Pipeline: unsupervised algorithm")
    print("50 readers input")
    print("Features: d2v, average word count, 3 4 5 grams")

    df = load_50_authors_data_sets_to_dict()
    labels = df['labels']

    # extract features from data set
    tf_idf_3_grams = Pipeline([
                ('extract', MetaStyleSelector("style_features_full_text_test_set.pkl")),
                ('sel', ItemSelector(key=features)),
                ('tf', TfidfVectorizer(max_features=1000,
                                       strip_accents='unicode', token_pattern=r'\w{1,}',
                                       ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                                       stop_words='english')),
                ('svd', TruncatedSVD(n_components=20))
    ])

    tf_idf_4_grams = Pipeline([
                ('extract', MetaStyleSelector("style_features_full_text_test_set.pkl")),
                ('sel', ItemSelector(key=features)),
                ('tf', TfidfVectorizer(max_features=1000,
                          strip_accents='unicode', token_pattern=r'\w{1,}',
                          ngram_range=(1, 4), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')),
                ('svd', TruncatedSVD(n_components=20))
            ])
    tf_idf_5_grams = Pipeline([
                ('sel', ItemSelector(key='text')),
                ('tf', TfidfVectorizer(max_features=1500,
                          strip_accents='unicode', token_pattern=r'\w{1,}',
                          ngram_range=(1, 5), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')),
                ('svd', TruncatedSVD(n_components=20))
            ])

    # average word count feature extraction pipeline
    word_count_pipeline = Pipeline([
                         ("word_count", AverageWordsSelector())])

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

    # Plot data in 2D using PCA (DIM reduction):
    from sklearn.decomposition import PCA as sklearnPCA
    pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
    x_transformed_2D = pd.DataFrame(pca.fit_transform(combined_features))
    for i in range(0, number_of_clusters):
        plt.scatter(x_transformed_2D[labels == i][0], x_transformed_2D[labels == i][1], c=np.random.rand(3, ),
                    label='author ' + str(i))
    plt.title('Sentence plot with DIM reduction (200D -> 2D)')
    plt.xlabel('X axis label')
    plt.ylabel('Y axis label')
    plt.legend()
    plt.show()  # need to manually close window
