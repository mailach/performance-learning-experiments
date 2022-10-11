import click
import os


import mlflow
import mlflow.sklearn

import pandas as pd
from sklearn import tree

import logging
from rich.logging import RichHandler


from utils.caching import CacheHandler

from learning.models import LearnerFactory, Learner


logging.basicConfig(
    level=logging.INFO,
    format="LEARNING    %(message)s",
    handlers=[RichHandler()],
)


def _predict_on_test(learner: Learner, test_X: pd.DataFrame, test_Y: pd.Series):
    pred = pd.Series(learner.predict(test_X))
    pred.name = "predicted"
    return pd.concat([pred, test_Y], axis=1)


def _load_data(data_file: str, cache: CacheHandler):
    data = pd.DataFrame(cache.retrieve(data_file))
    Y = data["measured_value"]
    X = data.drop("measured_value", axis=1)
    return X, Y


hyperparams = {
    "svr": ["epsilon", "coef0", "shrinking", "tol"],
    "rf": ["random_state", "max_features", "n_estimators", "min_samples_leaf"],
    "cart": ["min_samples_split", "min_samples_leaf"],
    "knn": ["n_neighbors", "weights", "algorithm", "p"],
    "krr": ["alpha", "kernel", "degree", "gamma"],
}


@click.command(
    help="Learn from sampled configurations",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option("--sampling_run_id")
@click.option("--method")
@click.option("--workflow_id")
@click.option("--min_samples_split", type=int, default=2)
@click.option("--min_samples_leaf", type=int, default=1)
@click.option("--c", type=float)
@click.option("--epsilon", type=float)
@click.option("--shrinking", type=bool)
@click.option("--tol", type=float)
@click.option("--n_neighbors", type=int)
@click.option("--weigths", type=str)
@click.option("--algorithm", type=str)
@click.option("--p", type=int)
@click.option("--kernel", type=str)
@click.option("--degree", type=float)
@click.option("--gamma", type=float, default=None)
@click.option("--alpha", type=float)
def learning(sampling_run_id: str, method: str, **kwargs):
    logging.info("Start learning from sampled configurations.")
    params = {k: v for k, v in kwargs.items() if k in hyperparams[method]}

    sampling_cache = CacheHandler(sampling_run_id)
    train_X, train_Y = _load_data("train.json", sampling_cache)
    test_X, test_Y = _load_data("test.json", sampling_cache)

    with mlflow.start_run() as run:
        model_cache = CacheHandler(run.info.run_id)
        learner = LearnerFactory(method)
        logging.info(f"Use hyperparameter: {params}")
        learner.set_parameters(params)
        learner.fit(train_X, train_Y)

        logging.info(f"Log model and save to cache {model_cache.cache_dir}")

        learner.log(model_cache.cache_dir)

        logging.info("Predict test set and save to cache.")
        prediction = _predict_on_test(learner, test_X, test_Y)
        model_cache.save({"predicted.json": prediction.to_dict("records")})
        mlflow.log_artifact(os.path.join(model_cache.cache_dir, "predicted.json"), "")


if __name__ == "__main__":
    learning()
