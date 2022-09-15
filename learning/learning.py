import pandas as pd
from sklearn import tree

# minbucket = minimum sample size for any leaf = min_samples_leaf
# minsplit = minimum sample size of a segment before it is used to further split = min_samples_split
# |S| = size of input sample, my is 100

# if |S| <= 100 -> minbucket = floor(|s|/10 + 1/2) & minsplit = 2 * minbucket
# else: minsplit = floor(|s|/10 + 1/2) and minbucket = floor(minsplit/2)
# minimum minbucket = 2
# minimum minsplit = 4

# min_samples_leaf = min_samples_leaf, min_samples_split=min_samples_split
def cart(train: pd.DataFrame, cart_arguments: dict[str, any] = {}) -> tree.DecisionTreeRegressor:

    Y = train["measured_value"]
    X = train.drop("measured_value", axis=1)

    model = tree.DecisionTreeRegressor()
    model.fit(X, Y, **cart_arguments)


    return model