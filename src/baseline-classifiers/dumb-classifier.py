from src.evaluations.evaluations import *
from src.utils.input_reader import *


# A dumb classifier to calculate log-loss of majority predictor on dataset
if __name__ == '__main__':
    # LOAD DATA
    path_prefix = ".." + os.sep + ".." + os.sep + "input" + os.sep
    train_df, test_df, sample_df = load_data_sets(path_prefix + "train.csv", path_prefix + "test.csv", None)
    labels = train_df['author_label']
    train_df['0'] = 1
    train_df['1'] = 0
    train_df['2'] = 0
    preds = train_df[['0','1','2']]
    print("logloss: %0.3f " % metrics.log_loss(labels, preds))
