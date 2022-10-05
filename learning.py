import click


import mlflow
import mlflow.sklearn

import pandas as pd
from sklearn import tree

import logging
from rich.logging import RichHandler


from utils.caching import CacheHandler

from learning.models import cart, LearnerFactory, Learner
from learning.metrics import prediction_fault_rate


logging.basicConfig(
    level=logging.INFO,
    format="LEARNING    %(message)s",
    handlers=[RichHandler()],
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


def _evaluate(model: Learner, test: pd.DataFrame) -> None:
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
@click.option("--sampling_run_id")
@click.option("--method")
@click.option("--min_samples_split", type=int, default=2)
@click.option("--min_samples_leaf", type=int, default=1)
def learning(
    sampling_run_id: str, method: str, min_samples_split: int, min_samples_leaf: int
):

    logging.info("Start learning from sampled configurations.")

    sampling_cache = CacheHandler(sampling_run_id)

    train = pd.DataFrame(sampling_cache.retrieve("sampled_configurations.json"))
    train_Y = train["measured_value"]
    train_X = train.drop("measured_value", axis=1)

    test = pd.DataFrame(sampling_cache.retrieve("remaining_configurations.json"))

    with mlflow.start_run() as run:
        model_cache = CacheHandler(run.info.run_id)
        learner = LearnerFactory(method)
        learner.set_parameters(
            {
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
            }
        )
        learner.fit(train_X, train_Y)
        logging.info(f"Log model to registry and save to cache {model_cache.cache_dir}")
        learner.log(model_cache.cache_dir)


if __name__ == "__main__":
    learning()
