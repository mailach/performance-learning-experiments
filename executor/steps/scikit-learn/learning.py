from caching import CacheHandler
from rich.logging import RichHandler
import pandas as pd
import mlflow.sklearn
import mlflow
import click
import logging
import time

import os


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, make_scorer

from joblib import parallel_backend


def activate_logging(logs_to_artifact):
    with open("logs.txt", "w"):
        pass
    if logs_to_artifact:
        return logging.basicConfig(
            filename="logs.txt",
            level=logging.INFO,
            format="LEARNING    %(message)s",
        )
    return logging.basicConfig(
        level=logging.INFO,
        format="LEARNING    %(message)s",
        handlers=[RichHandler()],
    )


def _make_pred_artifact(pred, test_y: pd.Series):
    prediction = pd.concat([pd.Series(pred), test_y], axis=1)
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


tuning_params = {
    "grid_search": {
        "svr": {
            "kernel": ["linear", "poly", "rbf"],
            "C": [0.001, 0.01, 0.1, 1, 10, 100, 500, 750, 250, 1000.0],
            "gamma": [0.001, 0.01, 0.1, 0.005, 0.05, 0.5, 0.25, 0.025, 0.075, 1.0],
            "epsilon": [0.001, 0.01, 0.1, 0.5, 1.0],
        },
        "cart": {
            "min_samples_split": [],
            "min_samples_leaf": [],
            "ccp_alpha": [0.000001, 0.00001, 0.0001, 0.001, 0.01],
            "random_state": [1],
        },
        "rf": {
            "max_features": [],
            "n_estimators": list(range(2, 30)),
        },
        "knn": {
            "n_neighbors": list(range(2, 21)),
            "weights": ["distance", "uniform"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "p": list(range(1, 6)),
        },
        "kr": {
            "kernel": ["poly", "rbf", "linear"],
            "alpha": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.05],
            "degree": list(range(1, 6)),
            "gamma": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.05],
        },
        "bagging": {
            "base_estimator": [
                DecisionTreeRegressor(min_samples_leaf=i) for i in range(1, 11)
            ],
            "n_estimators": list(range(2, 21)),
        },
    }
}


def create_param_grid(tuning_strategy, method, n_features):
    param_space = tuning_params[tuning_strategy][method]

    if method == "cart":
        param_space["min_samples_split"] = list(range(2, n_features))
        param_space["min_samples_leaf"] = [
            round(1 / 3 * minsplit) for minsplit in param_space["min_samples_split"]
        ]
    elif method == "rf":
        param_space["max_features"] = list(range(2, n_features))

    return param_space


model_selection = {"grid_search": GridSearchCV}


estimators = {
    "svr": SVR,
    "cart": DecisionTreeRegressor,
    "rf": RandomForestRegressor,
    "knn": KNeighborsRegressor,
    "kr": KernelRidge,
    "bagging": BaggingRegressor,
}


def MRE(y_true, y_pred):
    mre = mean_absolute_percentage_error(y_true, y_pred)
    return mre


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
@click.option("--tuning_strategy", type=str, default=None)
@click.option("--logs_to_artifact", type=bool, default=False)
def learning(
    sampling_run_id: str = "",
    method: str = "cart",
    nfp: str = "",
    tuning_strategy: str = None,
    logs_to_artifact: bool = False,
):
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
    activate_logging(logs_to_artifact)
    logging.info("Start learning from sampled configurations.")

    # load data
    sampling_cache = CacheHandler(sampling_run_id, new_run=False)
    train_x, train_y = _load_data("train.tsv", sampling_cache, nfp)
    test_x, test_y = _load_data("test.tsv", sampling_cache, nfp)

    # get model
    model = estimators[method]()

    # get parameter space
    param_space = create_param_grid(tuning_strategy, method, len(train_x.columns))

    # check if 10 features are available, elsewise use 9-fold cross validation
    k = 10 if len(train_x) > 10 else 9

    # generate experiment
    selection = model_selection[tuning_strategy](
        model,
        param_space,
        n_jobs=-1,
        verbose=1,
        cv=k,
        scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=False),
    )

    with mlflow.start_run() as run:
        try:
            model_cache = CacheHandler(run.info.run_id)
            logging.info("Start hyperparam search using: %s", str(param_space))
            start = time.perf_counter_ns()

            with parallel_backend("threading"):
                selection.fit(train_x, train_y)
            end = time.perf_counter_ns()
            mlflow.log_metric("learning_time", (end - start) * 0.000000001)
            mlflow.sklearn.log_model(selection.best_estimator_, "")
            mlflow.log_params(selection.best_params_)
            mlflow.log_metric("best_score", selection.best_score_)
            logging.info("Predict on test set and save to cache.")
            prediction = _make_pred_artifact(
                selection.best_estimator_.predict(test_x), test_y
            )
            model_cache.save({"predicted.tsv": prediction})
            mlflow.log_artifact(
                os.path.join(model_cache.cache_dir, "predicted.tsv"), ""
            )

        except Exception as e:
            logging.error("During learning the following error occured: %s", e)
            raise e
        finally:
            if logs_to_artifact:
                mlflow.log_artifact("logs.txt", "")


if __name__ == "__main__":
    # pylint: disable-next=no-value-for-parameter
    learning()
