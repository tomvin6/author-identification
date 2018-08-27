# this class is to produce all relevant inspected statistics on the data set
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import input_reader
from src.features.pos_tagging import *
from src.features.santimant_features import *
from src.features.writing_style_features import *
import os
import re
import nltk

nltk.download('maxent_ne_chunker')
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string


def seperate_entity_types(sentence):
    tokens = sentence.split()

    matcher = re.compile('^ent__')
    ent_types = ''
    for token in tokens:
        if matcher.match(token) is not None:
            ent_types = ent_types + token + " "

    exclude = set(string.punctuation)
    ent_types = ''.join(ch for ch in ent_types if ch not in exclude)
    ent_types = ent_types.replace('ent', '')
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
    if title_prefix != '':
        title = title_prefix + ',token frequencies, top 30'
    else:
        title = 'token frequencies, top 30'

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
    return s[(med - 3 * std <= s) & (s <= med + 3 * std)]


def plot_panctuation_use(author_df):
    f, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(drop_outliers(author_df.loc[author_df.author == 'EAP'].punct_cnt), shade=True, color="r");
    sns.kdeplot(drop_outliers(author_df.loc[author_df.author == 'HPL'].punct_cnt), shade=True, color="g");
    sns.kdeplot(drop_outliers(author_df.loc[author_df.author == 'MWS'].punct_cnt), shade=True, color="b");
    ax.legend(labels=['EAP', 'HPL', 'MWS']).set_title('panctuation use per author')
    f.savefig('output_charts//plot_punctuation_use.pdf', format='pdf')


def plot_noun_use(author_df):
    f, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(drop_outliers(author_df.loc[author_df.author == 'EAP'].noun_chunks_cnt), shade=True, color="r");
    sns.kdeplot(drop_outliers(author_df.loc[author_df.author == 'HPL'].noun_chunks_cnt), shade=True, color="g");
    sns.kdeplot(drop_outliers(author_df.loc[author_df.author == 'MWS'].noun_chunks_cnt), shade=True, color="b");
    ax.legend(labels=['EAP', 'HPL', 'MWS']).set_title('noun use per author')
    f.savefig('output_charts//plot_noun_use.pdf', format='pdf')


def plot_author_polarity(author_df):
    f, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(drop_outliers(author_df.loc[author_df.author == 'EAP'].polarity_of_text), shade=True, color="r");
    sns.kdeplot(drop_outliers(author_df.loc[author_df.author == 'HPL'].polarity_of_text), shade=True, color="g");
    sns.kdeplot(drop_outliers(author_df.loc[author_df.author == 'MWS'].polarity_of_text), shade=True, color="b");
    ax.legend(labels=['EAP', 'HPL', 'MWS']).set_title('text polarity per author')
    f.savefig('output_charts//plot_text_polarity.pdf', format='pdf')


if __name__ == '__main__':
    df_train = load_data()
    print(df_train.info())
    author_bar_plot(df_train)

    # remove stop words and punctuations
    preprocess_text(df_train)

    df_train['text_cleaned2'] = df_train['text_cleaned'].str.replace('-PRON-', '')
    # plot common words per author
    plot_token_frequencies(author_df=df_train, title_prefix='Overall word')
    df_eap = df_train.loc[df_train.author == 'EAP']
    plot_token_frequencies(author_df=df_eap, title_prefix='EAP word')
    df_hpl = df_train.loc[df_train.author == 'HPL']
    plot_token_frequencies(author_df=df_hpl, title_prefix='HPL word')
    df_mws = df_train.loc[df_train.author == 'MWS']
    plot_token_frequencies(author_df=df_mws, title_prefix='MWS word')

    # enteties
    df_train['entities_types'] = df_train.text_ent_repl.apply(seperate_entity_types)
    plot_token_frequencies(author_df=df_train, referance_column='entities_types', title_prefix='Overall entities')
    df_eap = df_train.loc[df_train.author == 'EAP']
    plot_token_frequencies(author_df=df_eap, referance_column='entities_types', title_prefix='EAP entities')
    df_hpl = df_train.loc[df_train.author == 'HPL']
    plot_token_frequencies(author_df=df_hpl, referance_column='entities_types', title_prefix='HPL entities')
    df_mws = df_train.loc[df_train.author == 'MWS']
    plot_token_frequencies(author_df=df_mws, referance_column='entities_types', title_prefix='MWS entities')

    # additional stats
    boxplot_length(df_train)
    boxplot_char_length(df_train)
    plot_panctuation_use(df_train)
    plot_noun_use(df_train)
    plot_author_polarity(df_train)
