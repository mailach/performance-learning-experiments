import logging
from abc import ABC, abstractmethod


import pandas as pd
import mlflow


from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt

from pim.learning.feature_selection import ForwardFeatureSelector


class Learner(ABC):
    """Abstract class for Learning"""

    model = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def log(self, cache_dir: str) -> None:
        pass


class ScikitLearner(Learner):
    feature_names = []

    def load(self, cache_dir: str) -> None:
        self.model = mlflow.sklearn.load_model(cache_dir)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X[self.feature_names])

    @abstractmethod
    def log(self, cache_dir: str) -> None:
        pass


# minbucket = minimum sample size for any leaf = min_samples_leaf
# minsplit = minimum sample size of a segment before it is used to further split = min_samples_split
# |S| = size of input sample, my is 100

# if |S| <= 100 -> minbucket = floor(|s|/10 + 1/2) & minsplit = 2 * minbucket
# else: minsplit = floor(|s|/10 + 1/2) and minbucket = floor(minsplit/2)
# minimum minbucket = 2
# minimum minsplit = 4

# min_samples_leaf = min_samples_leaf, min_samples_split=min_samples_split


class CARTLearner(ScikitLearner):
    def __init__(self, params: dict[str, any]) -> None:
        self.model = DecisionTreeRegressor(**params)

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        self.model.fit(X, Y)
        self.feature_names = X.columns

    def log(self, cache_dir) -> None:
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="",
            registered_model_name="CART",
        )
        mlflow.sklearn.save_model(sk_model=self.model, path=cache_dir)
        logging.info("Visualize tree")
        self._visualize()

    def _visualize(self) -> None:

        fig = plt.figure()
        tree.plot_tree(
            self.model,
            feature_names=self.feature_names,
            # class_names=iris.target_names,
            filled=True,
            fontsize=1,
        )

        mlflow.log_figure(figure=fig, artifact_file="decistion_tree.png")


# min samples leaf The minimal number of configurations required in each leaf node.
# random state The seed that is used in the random number generator.
# n estimators The number of trees in the forest.
# max features The number of features to consider when looking for the best split:


class RFLearner(ScikitLearner):
    def __init__(self, params: dict[str, any]) -> None:
        self.model = RandomForestRegressor(**params)

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        self.model.fit(X, Y)
        self.feature_names = X.columns

    def log(self, cache_dir) -> None:
        logging.info("Log model to registry and save to cache %s", cache_dir)
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="",
            registered_model_name="RF",
        )
        mlflow.sklearn.save_model(sk_model=self.model, path=cache_dir)


# c
# epsilon
# coef0
# shrinking
# tol
class SvrLearner(ScikitLearner):
    def __init__(self, params: dict[str, any]) -> None:
        self.model = SVR(**params)

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        self.model.fit(X, Y)
        self.feature_names = X.columns

    def log(self, cache_dir) -> None:
        logging.info("Log model to registry and save to cache %s", cache_dir)
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="",
            registered_model_name="SVR",
        )
        mlflow.sklearn.save_model(sk_model=self.model, path=cache_dir)


# alpha Parameter that aims at reducing the variance of the predictions.
# kernel Defines the kind of kernel being considered (e.g., linear).
# degree The degree of the polynomial kernel.
# gamma


class KrrLearner(ScikitLearner):
    def __init__(self, params: dict[str, any]) -> None:
        self.model = KernelRidge(**params)

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        self.model.fit(X, Y)
        self.feature_names = X.columns

    def log(self, cache_dir) -> None:
        logging.info("Log model to registry and save to cache %s", cache_dir)
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="",
            registered_model_name="KRR",
        )
        mlflow.sklearn.save_model(sk_model=self.model, path=cache_dir)


# n_neighbors Number of configurations considered in a prediction.
# weights Weights of the neighbor configurations.
# algorithm Algorithm used to compute the neighbors.
# p


class KnnLearner(ScikitLearner):
    def __init__(self, params: dict[str, any]) -> None:
        self.model = KNeighborsRegressor(**params)

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        self.model.fit(X, Y)
        self.feature_names = X.columns

    def log(self, cache_dir) -> None:
        logging.info("Log model to registry and save to cache %s", cache_dir)
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="",
            registered_model_name="KNN",
        )
        mlflow.sklearn.save_model(sk_model=self.model, path=cache_dir)


class MrLearner(ScikitLearner):
    def __init__(self, params: dict[str, any]) -> None:
        self.model = LinearRegression(**params)

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        selector = ForwardFeatureSelector(self.model)
        feature_set, _ = selector.select(X.copy(), Y)
        self.model.fit(feature_set, Y)
        self.feature_names = feature_set.columns

    def log(self, cache_dir) -> None:
        logging.info("Log model to registry and save to cache %s", cache_dir)
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="",
            registered_model_name="MR",
        )
        mlflow.sklearn.save_model(sk_model=self.model, path=cache_dir)


def LearnerFactory(method: str, params: dict[str, any]) -> Learner:

    learners = {
        "cart": CARTLearner,
        "rf": RFLearner,
        "mr": MrLearner,
        "svr": SvrLearner,
        "krr": KrrLearner,
        "knn": KnnLearner,
    }

    return learners[method](params)
