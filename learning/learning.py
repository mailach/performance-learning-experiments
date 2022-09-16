import logging
import json
import os
import click


import mlflow
import mlflow.sklearn

import pandas as pd
from sklearn import tree

from utils.models import cart
from utils.metrics import prediction_fault_rate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s Learning:  %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
)


# minbucket = minimum sample size for any leaf = min_samples_leaf
# minsplit = minimum sample size of a segment before it is used to further split = min_samples_split
# |S| = size of input sample, my is 100

# if |S| <= 100 -> minbucket = floor(|s|/10 + 1/2) & minsplit = 2 * minbucket
# else: minsplit = floor(|s|/10 + 1/2) and minbucket = floor(minsplit/2)
# minimum minbucket = 2
# minimum minsplit = 4

# min_samples_leaf = min_samples_leaf, min_samples_split=min_samples_split


# def _prediction_accuracy(Y, Y_hat):
#    return 1 - _prediction_fault_rate(Y, Y_hat)
def _log_metrics(vals: pd.Series, metric: str) -> None:
    mlflow.log_metrics(
        {
            f"mean_{metric}": vals.mean(),
            f"median_{metric}": vals.median(),
            f"sd_{metric}": vals.std(),
        }
    )


def _evaluate(model, test: pd.DataFrame) -> None:
    fr = prediction_fault_rate(
        test["measured_value"], model.predict(test.drop("measured_value", axis=1))
    )
    _log_metrics(fr, "fault_rate")


def _model(train: pd.DataFrame, method: str, cart_args: dict[str, int] = {}):
    if method == "cart":
        mlflow.log_params(cart_args)

        model = cart(train, cart_args)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="cart",
            registered_model_name="cart",
        )
        return model
    else:
        logging.error("Learning method not implemented yet.")
        raise NotImplementedError


@click.command(
    help="Learn from sampled configurations",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option("--sample_path")
@click.option("--method")
@click.option("--min_samples_split", type=int, default=2)
@click.option("--min_samples_leaf", type=int, default=1)
def learning(
    sample_path: str, method: str, min_samples_split: int, min_samples_leaf: int
):
    mlflow.set_experiment(method)

    logging.info("Start learning from sampled configurations.")
    with open(os.path.join(sample_path, "sampled_configurations.json"), "r") as f:
        train = pd.DataFrame(json.load(f))

    with open(os.path.join(sample_path, "remaining_configurations.json"), "r") as f:
        test = pd.DataFrame(json.load(f))

    with mlflow.start_run():
        model = _model(
            train,
            method,
            {
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
            },
        )

        _evaluate(model, test)


if __name__ == "__main__":
    learning()
