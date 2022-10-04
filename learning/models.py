import logging

import pandas as pd
from sklearn import tree


def cart(train: pd.DataFrame, cart_args: dict[str, int]):
    logging.info("Learning using CART.")

    Y = train["measured_value"]
    X = train.drop("measured_value", axis=1)

    model = tree.DecisionTreeRegressor(**cart_args)
    model.fit(X, Y)
    return model
