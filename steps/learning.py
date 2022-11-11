from pim.learning.models import LearnerFactory, Learner
from utils.caching import CacheHandler
from rich.logging import RichHandler
import pandas as pd
import mlflow.sklearn
import mlflow
import click
import logging
import os


logging.basicConfig(
    level=logging.INFO,
    format="LEARNING    %(message)s",
    handlers=[RichHandler()],
)


def _predict_on_test(learner: Learner, test_x: pd.DataFrame, test_y: pd.Series):
    pred = pd.Series(learner.predict(test_x))

    prediction = pd.concat([pred, test_y], axis=1)
    prediction.columns = ["predicted", "measured"]
    return prediction


def _load_data(data_file: str, cache: CacheHandler, nfp: str):
    data = cache.retrieve(data_file)
    nfp = "nfp_" + nfp
    columns_to_drop = [col for col in data.columns if "nfp_" in col and col != nfp]
    data = data.drop(columns_to_drop, axis=1)
    Y = data[nfp]
    X = data.drop(nfp, axis=1)
    return X, Y


hyperparams = {
    "svr": ["epsilon", "coef0", "shrinking", "tol"],
    "rf": ["random_state", "max_features", "n_estimators", "min_samples_leaf"],
    "cart": ["min_samples_split", "min_samples_leaf"],
    "knn": ["n_neighbors", "weights", "algorithm", "p"],
    "krr": ["alpha", "kernel", "degree", "gamma"],
    "mr": [],
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
@click.option("--nfp")
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
def learning(sampling_run_id: str = "", method: str = "cart", nfp: str = "", **kwargs):
    """
    Learning of influences of options on nfp

    Parameters
    ----------
    sampling_run_id : str
        run with sampled configurations as artifacts
    method : str
        learning method
    nfp : str
        name of nfp
    """
    logging.info("Start learning from sampled configurations.")
    params = {k: v for k, v in kwargs.items() if k in hyperparams[method]}

    sampling_cache = CacheHandler(sampling_run_id, new_run=False)
    train_x, train_y = _load_data("train.tsv", sampling_cache, nfp)
    test_x, test_y = _load_data("test.tsv", sampling_cache, nfp)

    with mlflow.start_run() as run:

        model_cache = CacheHandler(run.info.run_id)
        logging.info("Use hyperparameter: %s", params)
        learner = LearnerFactory(method, params)

        learner.fit(train_x, train_y)

        logging.info("Log model and save to cache %s", model_cache.cache_dir)

        learner.log(model_cache.cache_dir)
        logging.info("Predict test set and save to cache.")
        prediction = _predict_on_test(learner, test_x, test_y)

        model_cache.save({"predicted.tsv": prediction})
        mlflow.log_artifact(os.path.join(model_cache.cache_dir, "predicted.tsv"), "")


if __name__ == "__main__":
    learning()
