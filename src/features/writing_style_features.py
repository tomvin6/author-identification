import nltk


def char_count(sentence):
    """function to return number of chracters """
    return len(sentence['text'])


def word_count(sentence):
    text = sentence['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    return word_count


def unique_word_fraction(sentence):
    """function to calculate the fraction of unique words on total words of the text"""
    text = sentence['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    unique_count = list(set(text_splited)).__len__()
    return (unique_count / word_count)


def punctuations_fraction(sentence):
    """functiopn to claculate the fraction of punctuations over total number of characters for a given text """
    text = sentence['text']
    char_count = len(text)
    punctuation_count = len([c for c in text if c in string.punctuation])
    return (punctuation_count / char_count)


def fraction_noun(sentence):
    """function to give us fraction of noun over total words """
    text = sentence['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')])
    return (noun_count / word_count)


def fraction_adj(sentence):
    """function to give us fraction of adjectives over total words in given text"""
    text = sentence['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    adj_count = len([w for w in pos_list if w[1] in ('JJ', 'JJR', 'JJS')])
    return (adj_count / word_count)


def fraction_verbs(sentence):
    """function to give us fraction of verbs over total words in given text"""
    text = sentence['text']
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    word_count = text_splited.__len__()
    pos_list = nltk.pos_tag(text_splited)
    verbs_count = len([w for w in pos_list if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')])
    return (verbs_count / word_count)


def get_writing_style_features(train_df):
    train_df['unique_word_fraction'] = train_df.apply(lambda row: unique_word_fraction(row), axis=1)
    train_df['punctuations_fraction'] = train_df.apply(lambda row: punctuations_fraction(row), axis=1)
    train_df['char_count'] = train_df.apply(lambda row: char_count(row), axis=1)
    train_df['word_count'] = train_df.apply(lambda row: char_count(row), axis=1)
    train_df['fraction_noun'] = train_df.apply(lambda row: fraction_noun(row), axis=1)
    train_df['fraction_adj'] = train_df.apply(lambda row: fraction_adj(row), axis=1)
    train_df['fraction_verbs'] = train_df.apply(lambda row: fraction_verbs(row), axis=1)


    return train_df
