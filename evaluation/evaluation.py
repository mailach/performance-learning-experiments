import pandas as pd
from sklearn import tree


# Guo et al 2012
def prediction_fault_rate(Y, Y_hat):
    if any(Y==0):
        raise Exception("True value can not be zero.")
    return abs(Y - Y_hat) / Y


def prediction_accuracy(Y, Y_hat):
    return 1 - prediction_fault_rate(Y, Y_hat)