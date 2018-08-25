# this class is to produce all relevant inspected statistics on the data set
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import input_reader
import os
import re
import nltk
nltk.download('maxent_ne_chunker')
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string


clean_words=['a','']

def replace_ents(doc):
    prefix = 'ent__'
    text = str(doc.doc)
    for ent in doc.ents:
        text = text.replace(ent.orth_, prefix + ent.label_)
    return text

def seperate_entity_types(sentence):
    tokens = sentence.split()

    matcher = re.compile('^ent__')
    ent_types=''
    for token in tokens:
        if matcher.match(token) is not None:
            ent_types=ent_types+token+" "

    exclude = set(string.punctuation)
    ent_types = ''.join(ch for ch in ent_types if ch not in exclude)
    ent_types=ent_types.replace('ent','')
    return ent_types


def load_data():
    path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    train_df, test_df, sample_df = input_reader.load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv",
                                                               None)
    train_df = train_df.set_index('id')
    train_df.index = [id[2:] for id in train_df.index]
    return train_df


def author_bar_plot(train_df):
    fig = plt.figure()
    sns.countplot(x=train_df.author).set_title('records count per author')
    fig.show()
    fig.savefig('output_charts//author_bar_plot.pdf', format='pdf')

def plot_token_frequencies(author_df, referance_column='text_cleaned2', title_prefix=''):
    fig = plt.figure()
    fd = nltk.FreqDist([y for x in author_df[referance_column].str.split() for y in x])
    if title_prefix!='':
        title=title_prefix+',token frequencies, top 30'
    else:
        title='token frequencies, top 30'

    fd.plot(30, title=title)
    fig.savefig('output_charts//plot_token_frequencies_' + title_prefix + '.pdf', format='pdf')

def boxplot_length(author_df):
    fig = plt.figure()
    sns.boxplot(x='author', y='words_cnt', data=author_df).set_title('sentence words count per author')
    fig.show()
    fig.savefig('output_charts//sentence_words_boxplot.pdf', format='pdf')

def boxplot_char_length(author_df):
    fig = plt.figure()
    sns.boxplot(x='author', y='char_cnt', data=author_df).set_title('sentence characters per author')
    fig.show()
    fig.savefig('output_charts//sentence_chars_boxplot.pdf', format='pdf')

def drop_outliers(s):
    med = s.mean()
    std = s.std()
    return s[(med - 3*std <= s) & (s <= med + 3*std)]

def plot_panctuation_use(author_df):
    f, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(drop_outliers(author_df.loc[author_df.author == 'EAP'].punct_cnt), shade=True, color="r");
    sns.kdeplot(drop_outliers(author_df.loc[author_df.author == 'HPL'].punct_cnt), shade=True, color="g");
    sns.kdeplot(drop_outliers(author_df.loc[author_df.author == 'MWS'].punct_cnt), shade=True, color="b");
    ax.legend(labels=['EAP', 'HPL', 'MWS']).set_title('panctuation use per author')
    f.savefig('output_charts//plot_punctuation_use.pdf', format='pdf')


if __name__ == '__main__':
    nlp = spacy.load('en')
    STOP_WORDS.add("'s")
    STOP_WORDS.add("the")
    STOP_WORDS.add("a")
    for word in STOP_WORDS:
        lexeme = nlp.vocab[word]
        lexeme.is_stop = True


    df_train = load_data()
    doc = df_train.text.apply(nlp)
    print(df_train.info())

    #plot how many records we have per author in DB. random accuracy if we will return EAP constantly, will be 41%
    author_bar_plot(df_train)

    #remove stop words and punctuations
    clean_and_lemmatize = lambda x: ' '.join([t.lemma_ for t in x if not t.is_punct and not t.is_stop])
    df_train['text_cleaned'] = doc.apply(clean_and_lemmatize)
    df_train['text_cleaned2'] = df_train['text_cleaned'].str.replace('-PRON-', '')
    #plot common words per author
    plot_token_frequencies(author_df=df_train, title_prefix='Overall word')
    df_eap = df_train.loc[df_train.author == 'EAP']
    plot_token_frequencies(author_df=df_eap, title_prefix='EAP word')
    df_hpl = df_train.loc[df_train.author == 'HPL']
    plot_token_frequencies(author_df=df_hpl, title_prefix='HPL word')
    df_mws = df_train.loc[df_train.author == 'MWS']
    plot_token_frequencies(author_df=df_mws, title_prefix='MWS word')


    #enteties
    df_train['text_ent_repl'] = doc.apply(replace_ents)
    df_train['entities_types']= df_train.text_ent_repl.apply(seperate_entity_types)
    plot_token_frequencies(author_df=df_train,referance_column='entities_types',  title_prefix='Overall entities')
    df_eap = df_train.loc[df_train.author == 'EAP']
    plot_token_frequencies(author_df=df_eap,referance_column='entities_types',  title_prefix='EAP entities')
    df_hpl = df_train.loc[df_train.author == 'HPL']
    plot_token_frequencies(author_df=df_hpl,referance_column='entities_types',  title_prefix='HPL entities')
    df_mws = df_train.loc[df_train.author == 'MWS']
    plot_token_frequencies(author_df=df_mws, referance_column='entities_types', title_prefix='MWS entities')

    #noun_chunks
    df_train['text_noun_chunks']=doc.apply(lambda x: list(x.noun_chunks))

    df_train['punct_cnt'] = doc.apply(lambda x: len([t for t in x if t.is_punct]))
    df_train['words_cnt'] = doc.apply(lambda x: len([t for t in x if not t.is_punct]))
    df_train['char_cnt'] = df_train.text.apply(len)
    df_train['ents_cnt'] = doc.apply(lambda x: len(x.ents))
    df_train['noun_chunks_cnt'] = doc.apply(lambda x: len(list(x.noun_chunks)))

    #additional stats
    boxplot_length(df_train)
    boxplot_char_length(df_train)
    plot_panctuation_use(df_train)




#TODO- add plots:
#charts:
# number of charachters
# number of words
# number of panctuations
# use of nuons
# common enteties?!
# sentiment of author

