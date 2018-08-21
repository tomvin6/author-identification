
from nltk import pos_tag, word_tokenize
import pandas as pd


def pos_tag_sentence(sentence):
    sen_tokens = word_tokenize(sentence)
    sen_pos_tags = pos_tag(sen_tokens)
    pos_tags_only = [pos_tag_pair[1] for pos_tag_pair in sen_pos_tags]
    return " ".join(pos_tags_only)

def pos_tag_df(data):
    train_df_tmp = pd.DataFrame(data=data, columns=['text'])
    train_df_tmp.text=train_df_tmp.text.astype(str)
    train_df_tmp['pos_tag'] = train_df_tmp['text'].apply(lambda row: pos_tag_sentence(row))
    return_df= pd.DataFrame(columns=['text'])
    return_df['text']=train_df_tmp['pos_tag']
    return return_df
