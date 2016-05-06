from time import time

import numpy as np
import pandas as pd
import xgboost as xgb
from nolearn.lasagne import NeuralNet
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from helper import load_data
from net import scale_train_data, scale_test_data, make_net

# disable pandas warnings
pd.options.mode.chained_assignment = None
RANDOM_STATE = 42
TOTAL_NETS = 30


def process_clf(clf, clf_name, train_data, test_data, features, target, n_folds=5):
    neural_net = type(clf) is NeuralNet
    train_data['meta_' + clf_name] = np.zeros(len(train_data))
    test_data['meta_' + clf_name] = np.zeros(len(test_data))
    cross_validation_iterator = StratifiedKFold(train_data[target], n_folds=n_folds, shuffle=True, random_state=42)
    train_logloss = []
    valid_logloss = []
    if neural_net:
        x, y, encoder, scaler = scale_train_data(train_data[features], train_data[target])
        x_test, ids_test = scale_test_data(test_data[features], test_data.shot_id, scaler)
    else:
        x, y = train_data[features], train_data[target]
        x_test, ids_test = test_data[features], test_data.shot_id

    for train_indices, valid_indices in cross_validation_iterator:
        # split to train and valid sets
        if neural_net:
            x_train_cv = x[train_indices, :]
            y_train_cv = y[train_indices]
            x_valid_cv = x[valid_indices, :]
            y_valid_cv = y[valid_indices]
        else:
            x_train_cv = x.ix[train_indices, :]
            y_train_cv = y.iloc[train_indices]
            x_valid_cv = x.ix[valid_indices, :]
            y_valid_cv = y.iloc[valid_indices]

        # train learner
        clf.fit(x_train_cv, y_train_cv)

        # make predictions
        y_train_predicted = clf.predict_proba(x_train_cv)[:, 1]
        y_valid_predicted = clf.predict_proba(x_valid_cv)[:, 1]
        test_predicted = clf.predict_proba(x_test)[:, 1]
        train_data['meta_' + clf_name].iloc[valid_indices] = y_valid_predicted
        test_data['meta_' + clf_name] += test_predicted

        # store results
        train_logloss.append(log_loss(y_train_cv, y_train_predicted))
        valid_logloss.append(log_loss(y_valid_cv, y_valid_predicted))
    test_data['meta_' + clf_name] /= n_folds
    return np.mean(train_logloss), np.mean(valid_logloss)


def get_classifiers(n_features, ignored=None):
    classifiers = dict()
    classifiers['rf'] = RandomForestClassifier(n_estimators=250, criterion='gini', max_depth=6, min_samples_split=2,
                                               min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_features=0.7,
                                               max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=-1,
                                               random_state=RANDOM_STATE, verbose=0, warm_start=False,
                                               class_weight=None)
    classifiers['et'] = ExtraTreesClassifier(n_estimators=250, criterion='gini', max_depth=6, min_samples_split=2,
                                             min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_features=0.7,
                                             max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=-1,
                                             random_state=RANDOM_STATE, verbose=0, warm_start=False, class_weight=None)
    classifiers['xgb'] = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8, nthread=-1,
                                           learning_rate=0.01, max_depth=8, min_child_weight=1, n_estimators=600,
                                           objective='binary:logistic', seed=RANDOM_STATE, silent=True, subsample=0.8)
    classifiers['logreg'] = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=5.0, fit_intercept=True,
                                               intercept_scaling=1, class_weight=None, random_state=RANDOM_STATE,
                                               solver='lbfgs', max_iter=200, multi_class='ovr', verbose=0)
    classifiers['svc'] = SVC(C=5.0, kernel='rbf', degree=3, coef0=0.008, shrinking=True, probability=True,
                             tol=0.001, gamma='auto', cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                             random_state=RANDOM_STATE)
    classifiers['gbc'] = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=40, subsample=0.7,
                                                    min_samples_split=2, min_samples_leaf=5, max_depth=3, init=None,
                                                    min_weight_fraction_leaf=0.0, random_state=RANDOM_STATE, verbose=0,
                                                    max_features=None, max_leaf_nodes=None, warm_start=False)
    classifiers['sgd'] = SGDClassifier(loss='log', penalty='l2', alpha=0.01, l1_ratio=0.1, fit_intercept=True,
                                       n_iter=1000, shuffle=True, verbose=0, epsilon=0.1, n_jobs=-1,
                                       random_state=RANDOM_STATE, learning_rate='optimal', eta0=0.0, power_t=0.1,
                                       class_weight=None, warm_start=False, average=False)
    for n in [10, 20, 40, 80, 160, 320]:
        classifiers['knn' + str(n).zfill(3)] = KNeighborsClassifier(n_neighbors=n, weights='uniform', algorithm='auto',
                                                                    leaf_size=75, p=2, metric='minkowski',
                                                                    metric_params=None)
    for n in range(TOTAL_NETS):
        classifiers['net' + str(n).zfill(2)] = make_net(2, n_features, 100)

    for clf in ignored:
        del classifiers[clf]

    return classifiers


def get_meta_features():
    ignored = ['svc']
    train_data, test_data, features, target = load_data('data.csv', small=True, part=20)
    train_data = train_data.reset_index(drop=True)
    classifiers = get_classifiers(len(features), ignored=ignored)
    total = time()

    # get meta features
    for clf_name in sorted(classifiers):
        start_time = time()
        print('processing ' + clf_name)
        train_loss, valid_loss = process_clf(classifiers[clf_name], clf_name, train_data, test_data, features, target)
        passed = (time() - start_time) / 60
        print('total (train,valid) Log Loss = (%.5f,%.5f). took %.2f minutes' % (train_loss, valid_loss, passed))

    # average neural nets' outputs
    test_data['meta_net'] = np.zeros(len(test_data))
    train_data['meta_net'] = np.zeros(len(train_data))
    for n in range(TOTAL_NETS):
        col = 'meta_' + 'net' + str(n).zfill(2)
        test_data['meta_net'] += test_data[col]
        train_data['meta_net'] += train_data[col]
        test_data.drop(col, axis=1, inplace=True)
        train_data.drop(col, axis=1, inplace=True)
    test_data['meta_net'] /= TOTAL_NETS
    train_data['meta_net'] /= TOTAL_NETS

    # write to file
    train_data.to_csv('train_meta.csv', index=False)
    test_data.to_csv('test_meta.csv', index=False)
    print('Generating meta features took %.2f minutes' % ((time() - total) / 60))


if __name__ == '__main__':
    get_meta_features()
