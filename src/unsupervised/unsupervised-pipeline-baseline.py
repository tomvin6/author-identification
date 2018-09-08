from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from src.evaluations.evaluations import print_unsupervised_scores
from src.selectors.average_words_selector import MetaStyleSelector
from src.utils.input_reader import *

# Model parameters
number_of_clusters = 50  # clustering
features = ['polarity_of_text', 'punct_cnt', 'ents_cnt', 'noun_chunks_cnt', 'fraction_adj', 'fraction_verbs']

if __name__ == '__main__':
    print("Pipeline: Unsupervised algorithm baseline")
    print("50 readers test dataset")
    print('Features: %s' % ', '.join(map(str, features)))

    df = load_50_authors_data_sets_to_dict(train=False)
    labels = df['author_label']

    # extract features from data set
    feature_extraction_pipeline = Pipeline([
                ('extract', MetaStyleSelector("style_features_full_text_test_set.pkl"))])
    all_features = feature_extraction_pipeline.fit_transform(df)
    chosen_features = all_features[features]

    print("Running K-means on all features \n")
    km = KMeans(number_of_clusters, init='k-means++',
           max_iter=300, n_init=10, random_state=0)
    cluster_labels = pd.DataFrame(km.fit_predict(chosen_features, y=labels))

    print_unsupervised_scores(labels, cluster_labels)


