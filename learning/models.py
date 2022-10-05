import logging

import pandas as pd
import mlflow


from abc import ABC, abstractmethod

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


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


class CARTLearner(Learner):
    def set_parameters(self, params: dict[str, any]) -> None:
        self.model = DecisionTreeRegressor(**params)

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        self.model.fit(X, Y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

    def log(self, cache_dir) -> None:
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="",
            registered_model_name="CART",
        )
        mlflow.sklearn.save_model(sk_model=self.model, path=cache_dir)

    def load(self, cache_dir: str) -> None:
        self.model = mlflow.sklearn.load_model(cache_dir)


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


class SvmLearner(Learner):
    def __init__():
        raise NotImplementedError


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
        "svm": SvmLearner,
        "krr": KrrLearner,
        "knn": KnnLearner,
    }

    return learners[method]()


def cart(train: pd.DataFrame, cart_args: dict[str, int]):
    logging.info("Learning using CART.")

    Y = train["measured_value"]
    X = train.drop("measured_value", axis=1)

    model = DecisionTreeRegressor(**cart_args)
    model.fit(X, Y)
    return model
