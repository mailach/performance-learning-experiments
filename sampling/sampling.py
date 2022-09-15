import pandas as pd
from sklearn.model_selection import train_test_split


def true_random(data: pd.DataFrame, n_train: int, n_test: int):
    train, test = train_test_split(
        data, train_size=n_train, test_size=n_test, random_state=42
    )
    return train, test
