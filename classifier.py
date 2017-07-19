#! -*- coding:utf-8 -*-

from sklearn.cross_validation import cross_val_score

from collections import Counter

import numpy as np


def prepare_data(x, y, percentual_to_train):
    """Prepare data for training and validating with sklearn algoritmhs, dividing data in training/validating according
    to a given percentual_to_train

    Parameters
    ----------
    x : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape = [n_samples]
        Target values.

    percentual_to_train :
        Percentual used to break the data in training and validating

    Returns
    -------
    x_to_train : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Vectors which will be used as input for training sklearn algoritmhs
    y_to_train : array-like, shape = [n_samples]
        Target values which will be used as input for training sklearn algoritmhs
    x_to_validate : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Vectors which will be used as input for validating against sklearn algoritmhs
    y_to_validate : array-like, shape = [n_samples]
        Target values which will be used as input for validating against sklearn algoritmhs
    """
    total_to_train = int(round(percentual_to_train * len(x), 2))

    x_to_train = x[0:total_to_train]
    y_to_train = y[0:total_to_train]
    x_to_validate = x[total_to_train:]
    y_to_validate = y[total_to_train:]

    return x_to_train, y_to_train, x_to_validate, y_to_validate


def minimum_threshold(y_to_validate):
    """Returns a dummy prediction, where the algotmh would always use the most popular option. The one which happens
    most will be the minimum threshold (using the test data!)

    Parameters
    ----------
    y_to_validate : array-like, shape = [n_samples]
        Original target values.

    Returns
    -------
    minimum_threshold :

    """
    counter = max(Counter(y_to_validate).itervalues())
    minimum = (100.0 * counter / len(y_to_validate))
    return minimum


def fit_model(model, x_to_train, y_to_train, k):
    results = cross_val_score(model, x_to_train, y_to_train, cv=k)
    mean_results = np.mean(results) * 100.0
    return mean_results


def validate_model(model, x_to_train, y_to_train, x_to_validate, y_to_validate):
    model.fit(x_to_train, y_to_train)
    result = model.predict(x_to_validate)
    diff = (result == y_to_validate)

    points = sum(diff)
    result = 100.0 * points/len(y_to_validate)

    return result