import collections as col
import pandas as pd
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import train_test_split

# input_file is csv file with column id, text and author
# return data from with
def load_data_sets(train_path,test_path,sample_path,encode_lables=True):
    if train_path is None:
        train_path="input\\train.csv"
    if test_path is None:
        train_path="input\\test.csv"
    if sample_path is None:
        train_path="input\\sample_path.csv"

    train_df=pd.read_csv(train_path)
    test_df=pd.read_csv(test_path)
    sample_df=pd.read_csv(sample_path)

    if encode_lables:
        lbl_enc = preprocessing.LabelEncoder()
        train_df.author_label = lbl_enc.fit_transform(train_df.author.values)
    return train_df,test_df,sample_df

def train_vali_split(train_df):
    xtrain, xvalid, ytrain, yvalid = train_test_split(train_df.text.values, train_df.author_label,
                                                      stratify=train_df.author_label,
                                                      random_state=42,
                                                      test_size=0.1, shuffle=True)
    return xtrain, xvalid, ytrain, yvalid