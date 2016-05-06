import numpy as np
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne.base import TrainSplit
from nolearn.lasagne import NeuralNet

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def scale_train_data(features, target):
    x = features.values.copy().astype(np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(target).astype(np.int32)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x, y, encoder, scaler


def scale_test_data(features, ids, scaler):
    x = features.values.copy().astype(np.float32)
    ids = ids.astype(str)
    x = scaler.transform(x)
    return x, ids


class EarlyStopping(object):
    def __init__(self, patience=50):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_params = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_train = train_history[-1]['train_loss']
        current_epoch = train_history[-1]['epoch']

        if current_train > current_valid:
            return
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_params = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience <= current_epoch:
            if nn.verbose > 0:
                print('Early stopping')
                print('Best valid loss was {:.6f} at epoch {}'.format(self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_params)
            raise StopIteration()


def make_net(num_classes, num_features, max_epochs, patience=10, verbose=0):
    early_stopping = EarlyStopping(patience=patience)

    layers = [('input', InputLayer),
              ('dense1', DenseLayer),
              ('dropout1', DropoutLayer),
              ('dense2', DenseLayer),
              ('output', DenseLayer)]

    net = NeuralNet(layers=layers,

                    input_shape=(None, num_features),
                    dense1_num_units=200,
                    dropout1_p=0.5,
                    dense2_num_units=200,
                    output_num_units=num_classes,
                    output_nonlinearity=softmax,

                    update=nesterov_momentum,
                    update_learning_rate=0.01,
                    update_momentum=0.9,

                    on_epoch_finished=[early_stopping],

                    train_split=TrainSplit(eval_size=0.2),
                    verbose=verbose,
                    max_epochs=max_epochs)
    return net
