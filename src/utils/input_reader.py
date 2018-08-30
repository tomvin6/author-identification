import os
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# input_file is csv file with column id, text and author
# return data from with
def load_data_sets(train_path,test_path,sample_path,encode_lables=True):
    if train_path is None:
        train_path=  "input" + os.sep + "train.csv"
    if test_path is None:
        test_path= "input" + os.sep + "test.csv"
    if sample_path is None:
        sample_path= "input" + os.sep + "sample_path.csv"

    train_df=pd.read_csv(train_path)
    test_df=pd.read_csv(test_path)
    sample_df = None  # pd.read_csv(sample_path)

    if encode_lables:
        lbl_enc = preprocessing.LabelEncoder()
        train_df.author_label = lbl_enc.fit_transform(train_df.author.values)
        train_df['author_label'] = lbl_enc.fit_transform(train_df.author.values)
    return train_df,test_df,sample_df


def load_50_authors_data_sets_to_dict():
    print("Input data: 50 Authors data-set")
    root = ".." + os.sep + ".." + os.sep + '50-authors-input' + os.sep + 'C50train'
    labels = []
    docs = []
    for r, dirs, files in os.walk(root):
        for file in files:
            with open(os.path.join(r, file), "r") as f:
                docs.append(f.read())
            labels.append(r.replace(root, ''))
    data_dict = dict([('text', docs), ('labels', labels)])
    df = pd.DataFrame(data_dict)
    # encode labels
    le = preprocessing.LabelEncoder()
    df['labels'] = le.fit_transform(df.labels)
    df['id'] = df.index
    return df


def load_50_authors_data_sentences_to_dict():
    print("Input data: 50 Authors data-set")
    root = ".." + os.sep + ".." + os.sep + '50-authors-input' + os.sep + 'C50train'
    labels = []
    docs = []
    for r, dirs, files in os.walk(root):
        for file in files:
            with open(os.path.join(r, file), "r") as f:
                author_text=f.read()
                author_sentences=author_text.split(".\n")
                for sentence in author_sentences:
                    sentence=sentence.replace('\n','')
                    # print(sentence)
                    if sentence.__len__()>0 and sentence.count(' ')>1:
                        docs.append(sentence)
                        labels.append(r.replace(root, ''))
    data_dict = dict([('text', docs), ('author', labels)])
    df = pd.DataFrame(data_dict)
    # encode labels
    le = preprocessing.LabelEncoder()
    df['author_label'] = le.fit_transform(df.author)
    df['id'] = df.index
    return df

def load_50_authors_preprocessed_data():
    print("Input data: 50 Authors data-set")
    root = ".." + os.sep + ".." + os.sep + '50-authors-input' + os.sep + 'df_train_50_auth_preprocessed.tsv'
    df= pd.read_csv(root, sep='\t')
    return df


def load_txt_data_sets(train_path,test_path,sample_path,encode_lables=True):
    if train_path is None:
        train_path=  "input" + os.sep + "train.csv"
    if test_path is None:
        test_path= "input" + os.sep + "test.csv"
    if sample_path is None:
        sample_path= "input" + os.sep + "sample_path.csv"

    train_df=pd.read_table(train_path, sep="###",dtype=str, skipinitialspace=True, header=None)
    test_df=pd.read_table(test_path, sep="###",dtype=str, skipinitialspace=True, header=None)
    sample_df = None  # pd.read_csv(sample_path)

    if encode_lables:
        lbl_enc = preprocessing.LabelEncoder()
        train_df.author_label = lbl_enc.fit_transform(train_df[0])
        train_df['author_label'] = lbl_enc.fit_transform(train_df[0])
        test_df.author_label = lbl_enc.fit_transform(test_df[0])
        test_df['author_label'] = lbl_enc.fit_transform(test_df[0])
    return train_df,test_df,sample_df


def train_vali_split(train_df):
    # xtrain, xvalid, ytrain, yvalid = train_test_split(train_df.text.values, train_df.author_label,
    #                                                   stratify=train_df.author_label,
    #                                                   random_state=42,
    #                                                   test_size=0.2, shuffle=True)
    xtrain, xvalid, ytrain, yvalid = train_test_split(train_df.drop(['author','author_label'],axis=1), train_df.author_label,
                                                      stratify=train_df.author_label,
                                                      random_state=42,
                                                      test_size=0.2, shuffle=True)
    return xtrain, xvalid, ytrain, yvalid


def replace_tabs_at_front(filename, text_to_search, replacement_text):
    import fileinput
    with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace(text_to_search, replacement_text), end='')
