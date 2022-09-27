import logging
import pandas as pd


def prediction_fault_rate(Y: pd.Series, Y_hat: pd.Series):
    logging.info("Calculate prediction fault rate.")
    if any(Y == 0):
        raise Exception("True value can not be zero.")

    
    fr = abs(Y - Y_hat) / Y

    return fr
