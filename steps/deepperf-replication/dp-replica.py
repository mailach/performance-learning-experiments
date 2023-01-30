from abc import ABC

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


tf.random.set_seed(5)


def _get_error(y_pred, y_test):
    x = abs((pd.Series(y_test) - pd.Series(y_pred)) / pd.Series(y_test))
    return sum(x) / len(y_test)


def _scale_data(X, y, max_xy=None):
    if not max_xy:
        max_x = np.amax(X, axis=0)
        assert 0 not in max_x
        X = np.divide(X, max_x)

        max_y = np.max(y)
        assert max_y != 0
        y = np.divide(y, max_y)

        return X, y, max_x, max_y

    return (np.divide(X, max_xy[0]), np.divide(y, max_xy[1]))


def _get_hidden_layer(initialization):
    return tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_initializer=initialization,
    )


def _get_output_layer(initialization):
    return tf.keras.layers.Dense(
        1, activation="linear", kernel_initializer=initialization
    )


def _search_learning_rate(model, X_train, y_train, X_val, y_val):

    lr_range = np.logspace(np.log10(1e-10), np.log10(0.1), 10)
    results = {}

    for lr in lr_range:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
        model.fit(X_train, y_train, batch_size=len(X_train), epochs=2000)
        error = model.evaluate(X_val, y_val)
        results[lr] = error

    return min(results, key=results.get)


def _search_regularization_param(X_train, y_train, X_val, y_val, n_layers, lr):

    reg_params = np.logspace(np.log10(1e-10), np.log10(10), 30)
    results = {}

    for rp in reg_params:
        model = Sparse_model(n_layers, "glorot_normal", rp)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
        model.fit(X_train, y_train, batch_size=len(X_train), epochs=2000)
        error = model.evaluate(X_val, y_val)
        results[rp] = error

    return min(results, key=results.get)


def _search_layer(X_train, y_train, X_val, y_val, initialization, learning_rate):
    last_error = float("inf")
    best_n = 0

    for n in range(3, 21):
        model = Plain_model(n, initialization)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
        )
        model.fit(X_train, y_train, batch_size=len(X_train), epochs=2000)
        error = model.evaluate(X_val, y_val)
        if error < last_error:
            last_error = error
            best_n = n
        else:
            print(f"N layer: {best_n}, error: {last_error}")
            print(f"N+1 layer: {n}, error: {error}")
            return n


def _create_plain_model(layers, initialization):
    return tf.keras.models.Sequential(
        [_get_hidden_layer(initialization) for _ in range(layers)]
        + [_get_output_layer(initialization)]
    )


def _create_sparse_model(layers, initialization, regularization_parameter):
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                128,
                activation="relu",
                activity_regularizer=tf.keras.regularizers.L1(regularization_parameter),
                kernel_initializer=initialization,
            )
        ]
        + [_get_hidden_layer(initialization) for _ in range(layers)]
        + [_get_output_layer(initialization)],
    )


class Model(ABC):
    model = None

    def compile(self, optimizer, loss):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, X, y, batch_size, epochs):
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs)

    def evaluate(self, X, y):
        y_pred = pd.Series(self.model.predict(X).ravel())
        error = _get_error(y_pred, pd.Series(list(y)))
        return error


class Plain_model(Model):
    def __init__(self, n_layers, initialization):
        self.model = _create_plain_model(n_layers, initialization)


class Sparse_model(Model):
    def __init__(self, n_layers, initialization, regularization_parameter):
        self.model = _create_sparse_model(
            n_layers, initialization, regularization_parameter
        )


def _find_hyperparameters(X_train, y_train):

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.66)

    model = Plain_model(2, "glorot_normal")
    lr = _search_learning_rate(model, X_train, y_train, X_val, y_val)
    n_layers = _search_layer(X_train, y_train, X_val, y_val, "glorot_normal", lr)

    model = Plain_model(n_layers + 5, "glorot_normal")
    learning_rate = _search_learning_rate(model, X_train, y_train, X_val, y_val)

    regularization_parameter = _search_regularization_param(
        X_train, y_train, X_val, y_val, n_layers + 5, learning_rate
    )
    return (n_layers, learning_rate, regularization_parameter)


data = pd.read_csv("tmp-data.tsv", sep="\t")


labels = data.pop("nfp_Performance")

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, train_size=55, random_state=42
)


X_train, y_train, max_x, max_y = _scale_data(X_train, y_train)

n_layers, learning_rate, regularization_parameter = _find_hyperparameters(
    X_train, y_train
)


model = Sparse_model(n_layers + 5, "glorot_normal", regularization_parameter)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
)
model.fit(X_train, y_train, len(X_train), 2000)

X_test, y_test = _scale_data(X_test, y_test, (max_x, max_y))
error = model.evaluate(X_test, y_test)
