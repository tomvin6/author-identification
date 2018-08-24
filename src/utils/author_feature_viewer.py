from src.features.writing_style_features import *
from src.utils.input_reader import load_50_authors_data_sets_to_dict


if __name__ == '__main__':
    # calculate average words per author and normalize
    df = load_50_authors_data_sets_to_dict()
    grouped = df[['text', 'labels']].groupby('labels')
    df = grouped['text'].agg(lambda allItemsWithLabelX: allItemsWithLabelX.transform(word_count).mean())
    df_norm = (df - df.mean()) / (df.max() - df.min())