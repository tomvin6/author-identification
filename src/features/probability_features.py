from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import numpy as np

#this is to create a probability feature, according to input vectorizer, to the given train and test sets.
# can be used by NB classifier or logistic regression for example
def get_prob_vectorizer_features(xtrain, xtest,ytrain,ytest, vectorizer, col, model, prefix, cv=5):
    vectorizer.fit(xtrain[col].append(xtest[col]))
    X = vectorizer.transform(xtrain[col])
    y = ytrain
    X_test = vectorizer.transform(xtest[col])

    cv_scores = []
    pred_test = 0
    pred_train = np.zeros([xtrain.shape[0], len(set(ytrain))])
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=123)

    print('CV started')
    for train_index, dev_index in kf.split(X, y):
        X_train, X_dev = X[train_index], X[dev_index]
        y_train, y_dev = y[train_index], y[dev_index]

        model.fit(X_train, y_train)
        pred_dev = model.predict_proba(X_dev)
        pred_test += model.predict_proba(X_test)

        pred_train[dev_index, :] = pred_dev
        cv_scores.append(metrics.log_loss(y_dev, pred_dev))
        print('.', end='')

    print('')
    print("Mean CV LogLoss: %.3f" % (np.mean(cv_scores)))
    pred_test /= cv

    for i in range(len(set(ytrain))):
        xtrain[prefix + str(i)]=pred_train[:, i]

    for i in range(len(set(ytrain))):
        xtest[prefix + str(i)]=pred_test[:, i]

    #return the model to be used on a new row outside of db
    return model.fit(X, y)

    # xtrain[prefix + 'eap'] = pred_train[:, 0]
    # xtrain[prefix + 'hpl'] = pred_train[:, 1]
    # xtrain[prefix + 'mws'] = pred_train[:, 2]
    #
    # xtest[prefix + 'eap'] = pred_test[:, 0]
    # xtest[prefix + 'hpl'] = pred_test[:, 1]
    # xtest[prefix + 'mws'] = pred_test[:, 2]