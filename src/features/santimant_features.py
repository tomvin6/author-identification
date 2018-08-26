
from textblob import TextBlob
import pandas as pd

def get_polarity(sentence):
    sentiment = TextBlob(sentence)
    # polarity 1 for positive and 0 for negetive
    return sentiment.sentiment.polarity

def get_sentimant_features(train_df):
    train_df_tmp = pd.DataFrame(data=train_df, columns=['text'])
    train_df_tmp.text=train_df_tmp.text.astype(str)
    train_df_tmp['sentiment_polarity'] = train_df_tmp['text'].apply(lambda row: get_polarity(row))
    return train_df_tmp