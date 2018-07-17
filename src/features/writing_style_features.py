import nltk
import string
import pandas as pd

def char_count(sentence):
    """function to return number of chracters """
    return len(sentence)


def word_count(sentence):
    text_splited = sentence.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    return word_count


def unique_word_fraction(sentence):
    """function to calculate the fraction of unique words on total words of the text"""
    text_splited = sentence.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    unique_count = list(set(text_splited)).__len__()
    return (unique_count / word_count)


def punctuations_fraction(sentence):
    """functiopn to claculate the fraction of punctuations over total number of characters for a given text """
    char_count = len(sentence)
    punctuation_count = len([c for c in sentence if c in string.punctuation])
    return (punctuation_count / char_count)


def fraction_noun(sentence):
    """function to give us fraction of noun over total words """
    text_splited = sentence.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')])
    return (noun_count / word_count)


def fraction_adj(sentence):
    """function to give us fraction of adjectives over total words in given text"""
    text_splited = sentence.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    adj_count = len([w for w in pos_list if w[1] in ('JJ', 'JJR', 'JJS')])
    return (adj_count / word_count)


def fraction_verbs(sentence):
    """function to give us fraction of verbs over total words in given text"""
    text_splited = sentence.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    verbs_count = len([w for w in pos_list if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')])
    return (verbs_count / word_count)


def get_writing_style_features(train_df):
    train_df = pd.DataFrame(data=train_df, columns=['text'])
    train_df.text=train_df.text.astype(str)
    train_df['unique_word_fraction'] = train_df['text'].apply(lambda row: unique_word_fraction(row))
    train_df['punctuations_fraction'] = train_df['text'].apply(lambda row: punctuations_fraction(row))
    train_df['char_count'] = train_df['text'].apply(lambda row: char_count(row))
    train_df['word_count'] = train_df['text'].apply(lambda row: char_count(row))
    train_df['fraction_noun'] = train_df['text'].apply(lambda row: fraction_noun(row))
    train_df['fraction_adj'] = train_df['text'].apply(lambda row: fraction_adj(row))
    train_df['fraction_verbs'] = train_df['text'].apply(lambda row: fraction_verbs(row))

    return train_df
