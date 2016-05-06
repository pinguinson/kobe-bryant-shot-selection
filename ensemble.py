from nolearn.lasagne import NeuralNet

from helper import load_meta
from time import time

import numpy as np
import xgboost as xgb

from net import make_net, scale_test_data, scale_train_data
from helper import make_submission


def process_clf(clf, train_data, test_data, features, target):
    neural_net = type(clf) is NeuralNet
    if neural_net:
        x, y, encoder, scaler = scale_train_data(train_data[features], train_data[target])
        x_test, ids_test = scale_test_data(test_data[features], test_data.shot_id, scaler)
    else:
        x, y = train_data[features], train_data[target]
        x_test, ids_test = test_data[features], test_data.shot_id

    clf.fit(x, y)
    return clf.predict_proba(x_test)[:, 1]


def playground():
    train, test = load_meta('train_meta.csv', 'test_meta.csv')
    features = test.columns.tolist()
    features.remove('shot_id')
    target = 'shot_made_flag'
    print('processing nn ensemble')
    predicted = np.zeros(len(test))
    times = 100
    for i in range(times):
        print(i)
        start_time = time()
        clf = make_net(2, len(features), 100, verbose=0)
        predicted += process_clf(clf, train, test, features, target)
        print('took %.2f minutes' % ((time() - start_time) / 60))
    predicted /= times
    make_submission(predicted, test.shot_id, 'nn_' + str(times) + '.csv')


if __name__ == '__main__':
    playground()
