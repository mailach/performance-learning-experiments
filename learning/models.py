import logging
import os

import pandas as pd
import mlflow


from abc import ABC, abstractmethod

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from matplotlib import pyplot as plt


class Learner(ABC):
    @abstractmethod
    def set_parameters(self, params: dict[str, any]):
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass

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


class CARTLearner(Learner):
    def set_parameters(self, params: dict[str, any]) -> None:
        self.model = DecisionTreeRegressor(**params)

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        self.model.fit(X, Y)
        self.feature_names = X.columns

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

    def log(self, cache_dir) -> None:
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="",
            registered_model_name="CART",
        )
        mlflow.sklearn.save_model(sk_model=self.model, path=cache_dir)
        logging.info("Visualize tree")
        self._visualize(cache_dir)

    def load(self, cache_dir: str) -> None:
        self.model = mlflow.sklearn.load_model(cache_dir)

    def _visualize(self, cache_dir) -> None:
        fig = plt.figure(figsize=(25, 20))
        tree.plot_tree(
            self.model,
            feature_names=self.feature_names,
            # class_names=iris.target_names,
            filled=True,
        )

        mlflow.log_figure(figure=fig, artifact_file="decistion_tree.png")


# min samples leaf The minimal number of configurations required in each leaf node.
# random state The seed that is used in the random number generator.
# n estimators The number of trees in the forest.
# max features The number of features to consider when looking for the best split:


class RFLearner(Learner):
    def set_parameters(self, params: dict[str, any]) -> None:
        self.model = RandomForestRegressor(**params)

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        self.model.fit(X, Y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

    def log(self, cache_dir) -> None:
        logging.info(f"Log model to registry and save to cache {cache_dir}")
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="",
            registered_model_name="RF",
        )
        mlflow.sklearn.save_model(sk_model=self.model, path=cache_dir)

    def load(self, cache_dir: str) -> None:
        self.model = mlflow.sklearn.load_model(cache_dir)


# c
# epsilon
# coef0
# shrinking
# tol
class SvrLearner(Learner):
    def set_parameters(self, params: dict[str, any]) -> None:
        self.model = SVR(**params)

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        self.model.fit(X, Y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

    def log(self, cache_dir) -> None:
        logging.info(f"Log model to registry and save to cache {cache_dir}")
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="",
            registered_model_name="SVR",
        )
        mlflow.sklearn.save_model(sk_model=self.model, path=cache_dir)

    def load(self, cache_dir: str) -> None:
        self.model = mlflow.sklearn.load_model(cache_dir)


class KrrLearner(Learner):
    def __init__():
        raise NotImplementedError


class KnnLearner(Learner):
    def __init__():
        raise NotImplementedError


class MrLearner(Learner):
    def __init__():
        raise NotImplementedError


def LearnerFactory(method: str) -> Learner:

    learners = {
        "cart": CARTLearner,
        "rf": RFLearner,
        "mr": MrLearner,
        "svr": SvrLearner,
        "krr": KrrLearner,
        "knn": KnnLearner,
    }

    return learners[method]()
