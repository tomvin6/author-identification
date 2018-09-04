# this class is to produce all relevant inspected statistics on the data set
import os
import re
from collections import Counter

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nltk
import seaborn as sns

from src.features.writing_style_features import *
from src.utils import input_reader
from src.utils.input_reader import load_50_authors_preprocessed_data

nltk.download('maxent_ne_chunker')
import string
import os.path
import sys

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


def load_50_auth_data(train=True):
    train_df = input_reader.load_50_authors_data_sentences_to_dict(train=train)
    train_df = train_df.set_index('id')
    # train_df.index = [id[2:] for id in train_df.index]
    return train_df


def author_bar_plot(train_df):
    fig = plt.figure()
    sns.countplot(x=train_df.author, order=train_df.author.value_counts().index).set_title(
        'records count per author')
    plt.xticks(rotation='vertical', fontsize=5)
    fig.show()
    fig.tight_layout()
    fig.savefig('output_charts//author_bar_plot.pdf', format='pdf')
    print("author_bar_plot saved to output_charts//author_bar_plot.pdf")
    plt.close()


def plot_token_frequencies(author_df, referance_column='text_cleaned2', title_prefix='',
                           output_path='output_charts//50_authors_db//'):
    fig = plt.figure()
    fd = nltk.FreqDist([y for x in author_df[referance_column].str.split() for y in x])
    if title_prefix != '':
        title = title_prefix + ',token frequencies, top 30'
    else:
        title = 'token frequencies, top 30'

    fd.plot(30, title=title)
    fig.savefig(output_path + 'plot_token_frequencies_' + title_prefix + '.pdf', format='pdf')
    print("common words/entities for author "+ title_prefix+ " saved to "+output_path + 'plot_token_frequencies_' + title_prefix + '.pdf')
    plt.close()



def plot_multiple_token_frequencies(author_df, output_file_prefix, referance_column='text_cleaned2',
                                    referance_author_column='author'):
    fig = plt.figure()
    # fig.subplots_adjust(hspace=0.4, wspace=0.4,figsize=(15,15))
    gs1 = gridspec.GridSpec(10,5)
    gs1.tight_layout(fig)
    # fig.set_title('token frequencies, top 30')
    plt_i = 0
    for label in set(author_df[referance_author_column]):
        s_plt = fig.add_subplot(gs1[plt_i])
        plt_i = plt_i + 1
        s_plt.set_title(label)

        df = author_df[author_df[referance_author_column] == label]
        counter = Counter([y for x in df[referance_column].str.split() for y in x])
        top30 = counter.most_common(30)
        words = []
        words_count = []
        for i in range(0, len(top30)):
            str = top30[i][0]
            val = top30[i][1]
            words.append(str)
            words_count.append(val)
        width = 0.7
        plt.xticks(rotation='vertical', fontsize=5)
        plt.bar(words, words_count, width)
    fig.savefig('output_charts//50_authors_db//' + output_file_prefix + '.pdf', format='pdf')


def boxplot_length(author_df, out_folder=''):
    fig = plt.figure()
    sns.boxplot(x='author', y='words_cnt', data=author_df).set_title('sentence words count per author')
    plt.xticks(rotation='vertical', fontsize=5)
    fig.show()
    fig.savefig(out_folder + 'sentence_words_boxplot.pdf', format='pdf')
    print("boxplot for sentences word count saved to "+out_folder + 'sentence_words_boxplot.pdf')
    plt.close()


def boxplot_char_length(author_df, out_folder=''):
    fig = plt.figure()
    sns.boxplot(x='author', y='char_cnt', data=author_df).set_title('sentence characters per author')
    plt.xticks(rotation='vertical', fontsize=5)
    fig.show()
    fig.savefig(out_folder + 'sentence_chars_boxplot.pdf', format='pdf')
    print("boxplot for sentences length saved to " + out_folder + 'sentence_chars_boxplot.pdf')
    plt.close()


def drop_outliers(s):
    med = s.mean()
    std = s.std()
    return s[(med - 3 * std <= s) & (s <= med + 3 * std)]


def plot_panctuation_use(author_df, out_folder=''):
    f, ax = plt.subplots(figsize=(7, 4))
    for label in set(author_df.author):
        sns.kdeplot(drop_outliers(author_df.loc[author_df.author == label].punct_cnt), shade=True);
    ax.legend(labels=set(author_df.author)).set_title('panctuation use per author')
    f.savefig(out_folder + 'plot_punctuation_use.pdf', format='pdf')
    print("plot for panctuation use saved to " + out_folder + 'plot_punctuation_use.pdf')
    plt.close()


def plot_noun_use(author_df, out_folder=''):
    f, ax = plt.subplots(figsize=(7, 4))
    for label in set(author_df.author):
        sns.kdeplot(drop_outliers(author_df.loc[author_df.author == label].noun_chunks_cnt), shade=True);
    ax.legend(labels=set(author_df.author)).set_title('noun use per author')
    f.savefig(out_folder + 'plot_noun_use.pdf', format='pdf')
    print("plot for noun use saved to " + out_folder + 'plot_noun_use.pdf')
    plt.close()


def plot_author_polarity(author_df, out_folder=''):
    f, ax = plt.subplots(figsize=(7, 4))
    for label in set(author_df.author):
        sns.kdeplot(drop_outliers(author_df.loc[author_df.author == label].polarity_of_text), shade=True);
    ax.legend(labels=set(author_df.author)).set_title('text polarity per author')
    f.savefig('output_charts//' + out_folder + 'plot_text_polarity.pdf', format='pdf')
    print("plot for author polarity use saved to " + 'output_charts//' + out_folder + 'plot_text_polarity.pdf')
    plt.close()


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

#input args: path for external input data
if __name__ == '__main__':
    if len(sys.argv)>1:
        print("reading from external data file")
        input_data_path = sys.argv[1]
        df_train=pd.read_csv(input_data_path)
        print("preprocess data")
        preprocess_text(df_train)
    else:
        df_train = load_50_authors_preprocessed_data()

    print("data information: ")
    print(df_train.info())

    author_bar_plot(df_train)
    df_train['text_cleaned2'] = df_train['text_cleaned'].str.replace('-PRON-', '')

    # plot common words per author
    print("plotting common words per author:")
    for label in set(df_train['author']):
        df_label = df_train[df_train['author'] == label]
        author_name = label.replace('\\', '')
        plot_token_frequencies(author_df=df_label, title_prefix=author_name + '_word',
                               output_path='output_charts//word_frquencies//')

    #  plot common enteties per author
    print("plotting common entities per author:")
    df_train['entities_types'] = df_train.text_with_entities.apply(seperate_entity_types, "plot_enteties_freq_per_author")
    for label in set(df_train['author']):
        df_label = df_train[df_train['author'] == label]
        author_name = label.replace('\\', '')
        plot_token_frequencies(author_df=df_label, referance_column='entities_types',
                               title_prefix=author_name + '_entities',
                               output_path='output_charts//entities_frequencies//')

    # additional stats
    print("plotting additional stats:")
    boxplot_length(df_train, out_folder='output_charts//')
    df_train['char_cnt'] = df_train.text.apply(len)
    boxplot_char_length(df_train, out_folder='output_charts//')
    plot_panctuation_use(df_train, out_folder='output_charts//')
    plot_noun_use(df_train, out_folder='output_charts//')
    plot_author_polarity(df_train, out_folder='output_charts//')
