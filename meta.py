from time import time

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.cross_validation import cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from loader import load_data


def process_clf(clf, clf_name, train_data, test_data, features, target, n_folds=5):
    train_data['meta_' + clf_name] = np.zeros(len(train_data))
    test_data['meta_' + clf_name] = np.zeros(len(test_data))
    cross_validation_iterator = StratifiedKFold(train_data[target], n_folds=n_folds, shuffle=True, random_state=42)
    train_logloss = []
    valid_logloss = []

    for train_indices, valid_indices in cross_validation_iterator:
        # split to train and valid sets
        x_train_cv = train_data[features].ix[train_indices, :]
        y_train_cv = train_data[target].iloc[train_indices]
        x_valid_cv = train_data[features].ix[valid_indices, :]
        y_valid_cv = train_data[target].iloc[valid_indices]

        # train learner
        # print(x_train_cv.head())
        # print(y_train_cv.head())
        clf.fit(x_train_cv, y_train_cv)

        # make predictions
        y_train_predicted = clf.predict_proba(x_train_cv)[:, 1]
        y_valid_predicted = clf.predict_proba(x_valid_cv)[:, 1]
        test_predicted = clf.predict_proba(test_data[features])[:, 1]
        train_data['meta_' + clf_name].iloc[valid_indices] = y_valid_predicted
        test_data['meta_' + clf_name] += test_predicted

        # store results
        train_logloss.append(log_loss(y_train_cv, y_train_predicted))
        valid_logloss.append(log_loss(y_valid_cv, y_valid_predicted))
    test_data['meta_' + clf_name] /= n_folds
    return np.mean(train_logloss), np.mean(valid_logloss)


def playground():
    train_data, test_data, features, target = load_data('data.csv')
    train_data = train_data.reset_index(drop=True)
    clf = xgb.XGBClassifier()
    classificators = [(RandomForestClassifier(n_estimators=300, n_jobs=-1), 'rf'),
                      (ExtraTreesClassifier(n_estimators=300, n_jobs=-1), 'et'),
                      (xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1,
                                         colsample_bytree=0.8, learning_rate=0.01,
                                         max_depth=8, min_child_weight=1, n_estimators=1000,
                                         nthread=-1, objective='binary:logistic', seed=27,
                                         silent=False, subsample=0.8), 'xgb')]
    print(process_clf(clf, 'xgb', train_data, test_data, features, target))


if __name__ == '__main__':
    playground()
