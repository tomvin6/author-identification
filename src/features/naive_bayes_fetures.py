import numpy as np
from sklearn import metrics, model_selection, naive_bayes

from src.baseline_classifiers import naive_bayes
from src.features import tf_idf_features
import pandas as pd


# Y values should be encoded to numbers 0,1,2
def get_nb_features(xtrain, ytrain, xtest, lbl_prefix):
    clf = naive_bayes.NB_classifier()

    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([xtrain.shape[0], len(set(ytrain))])

    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(xtrain):
        dev_X, val_X = xtrain[dev_index], xtrain[val_index]
        dev_y, val_y = ytrain[dev_index], ytrain[val_index]

        clf.train(dev_X, dev_y)
        pred_val_y, cls_val_y = clf.predict(val_X)
        pred_test_y, cls_test_y = clf.predict(xtest)

        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index, :] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test / 5.

    columns = [lbl_prefix + str(i) for i in range(len(set(ytrain)))]
    return pd.DataFrame(columns=columns, data=pred_train), pd.DataFrame(columns=columns, data=pred_full_test)

    # # add the predictions as new features #
    # train_df["nb_cvec_eap"] = pred_train[:, 0]
    # train_df["nb_cvec_hpl"] = pred_train[:, 1]
    # train_df["nb_cvec_mws"] = pred_train[:, 2]
    #
    # test_df["nb_cvec_eap"] = pred_full_test[:, 0]
    # test_df["nb_cvec_hpl"] = pred_full_test[:, 1]
    # test_df["nb_cvec_mws"] = pred_full_test[:, 2]
