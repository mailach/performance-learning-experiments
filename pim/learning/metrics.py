import pandas as pd


def prediction_fault_rate(y: pd.Series, y_hat: pd.Series):
    """
    Calculates prediction fault rate.

    Parameters
    ----------
    y : pd.DataFrame
        measured value
    y_hat: pd.DataFrame
        predicted value
    """
    if any(y == 0):
        raise Exception("True value can not be zero.")

    fault_rate = abs(y - y_hat) / y

    return {
        "mean_fault_rate": fault_rate.mean(),
        "median_fault_rate": fault_rate.median(),
        "sd_fault_rate": fault_rate.std(),
    }
