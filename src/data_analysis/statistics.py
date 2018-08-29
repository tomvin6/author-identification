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
from collections import Counter
nltk.download('maxent_ne_chunker')
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
import numpy as np

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


def load_data(input_folder_name='input'):
    path_prefix = ".." + os.sep + ".." + os.sep + input_folder_name + os.sep
    train_df, test_df, sample_df = input_reader.load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv",
                                                               None)
    train_df = train_df.set_index('id')
    train_df.index = [id[2:] for id in train_df.index]
    return train_df

def load_50_auth_data():
    train_df = input_reader.load_50_authors_data_sentences_to_dict()
    train_df = train_df.set_index('id')
    # train_df.index = [id[2:] for id in train_df.index]
    return train_df


def author_bar_plot(train_df):
    fig = plt.figure()
    sns.countplot(x=train_df.author_label,order = train_df.author_label.value_counts().index).set_title('records count per author')
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

def plot_multiple_token_frequencies(author_df,output_file_prefix, referance_column='text_cleaned2',referance_author_column='author'):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.set_title('token frequencies, top 30')
    for label in set(author_df[referance_author_column]):
        s_plt= fig.add_subplot()
        df =author_df[author_df[referance_author_column]== label]
        counter = Counter([y for x in df[referance_column].str.split() for y in x])
        words = counter.keys()
        words_count = counter.values()

        indexes = np.arange(len(words))
        width = 0.7
        s_plt.bar(indexes, words_count, width)
        s_plt.xticks(indexes + width * 0.5, words)
        # plt.show()

        # fd = nltk.FreqDist([y for x in df[referance_column].str.split() for y in x])
        # fd.plot(30)
    fig.savefig('output_charts//50_authors_db//'+output_file_prefix + '.pdf', format='pdf')

def boxplot_length(author_df,out_sub_folder=''):
    fig = plt.figure()
    sns.boxplot(x='author', y='words_cnt', data=author_df).set_title('sentence words count per author')
    fig.show()
    fig.savefig('output_charts//sentence_words_boxplot.pdf', format='pdf')


def boxplot_char_length(author_df,out_sub_folder=''):
    fig = plt.figure()
    sns.boxplot(x='author', y='char_cnt', data=author_df).set_title('sentence characters per author')
    fig.show()
    fig.savefig('output_charts//sentence_chars_boxplot.pdf', format='pdf')


def drop_outliers(s):
    med = s.mean()
    std = s.std()
    return s[(med - 3 * std <= s) & (s <= med + 3 * std)]


def plot_panctuation_use(author_df,out_sub_folder=''):
    f, ax = plt.subplots(figsize=(7, 4))
    for label in set(author_df.author):
        sns.kdeplot(drop_outliers(author_df.loc[author_df.author == label].punct_cnt), shade=True);
    ax.legend(labels=set(author_df.author)).set_title('panctuation use per author')
    f.savefig('output_charts//'+out_sub_folder+'plot_punctuation_use.pdf', format='pdf')


def plot_noun_use(author_df,out_sub_folder=''):
    f, ax = plt.subplots(figsize=(7, 4))
    for label in set(author_df.author):
        sns.kdeplot(drop_outliers(author_df.loc[author_df.author == label].noun_chunks_cnt), shade=True);
    ax.legend(labels=set(author_df.author)).set_title('noun use per author')
    f.savefig('output_charts//'+out_sub_folder+'plot_noun_use.pdf', format='pdf')


def plot_author_polarity(author_df,out_sub_folder=''):
    f, ax = plt.subplots(figsize=(7, 4))
    for label in set(author_df.author):
        sns.kdeplot(drop_outliers(author_df.loc[author_df.author == label].polarity_of_text), shade=True);
    ax.legend(labels=set(author_df.author)).set_title('text polarity per author')
    f.savefig('output_charts//'+out_sub_folder+'plot_text_polarity.pdf', format='pdf')

def spooky_author_stats():
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

if __name__ == '__main__':
    df_train = load_50_auth_data()
    print(df_train.info())
    author_bar_plot(df_train)

    # remove stop words and punctuations
    preprocess_text(df_train)
    df_train['text_cleaned2'] = df_train['text_cleaned'].str.replace('-PRON-', '')
    # plot common words per author

    #TODO-test
    plot_multiple_token_frequencies(df_train,"plot_word_frequencies_per_author")

    # enteties
    df_train['entities_types'] = df_train.text_ent_repl.apply(seperate_entity_types,"plot_enteties_freq_per_author")
    plot_multiple_token_frequencies(author_df=df_train,referance_column="entities_types")

    # additional stats
    boxplot_length(df_train,out_sub_folder='50_authors_db//')
    boxplot_char_length(df_train,out_sub_folder='50_authors_db//')
    plot_panctuation_use(df_train,out_sub_folder='50_authors_db//')
    plot_noun_use(df_train,out_sub_folder='50_authors_db//')
    plot_author_polarity(df_train,out_sub_folder='50_authors_db//')
