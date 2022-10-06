import click
import os


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

def _predict_on_test(learner: Learner, test: pd.DataFrame):
    pred = pd.Series(learner.predict(test.drop("measured_value", axis=1)))
    pred.name = "predicted"
    return pd.concat([pred, test["measured_value"]], axis=1)


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
def learning(
    sampling_run_id: str, workflow_id: str, method: str, min_samples_split: int, min_samples_leaf: int
):

    logging.info("Start learning from sampled configurations.")

    sampling_cache = CacheHandler(sampling_run_id)

    train = pd.DataFrame(sampling_cache.retrieve(
        "sampled_configurations.json"))
    train_Y = train["measured_value"]
    train_X = train.drop("measured_value", axis=1)

    test = pd.DataFrame(sampling_cache.retrieve(
        "remaining_configurations.json"))

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

        logging.info(
            f"Log model and save to cache {model_cache.cache_dir}")

        learner.log(model_cache.cache_dir)

        logging.info("Predict test set and save to cache.")
        prediction = _predict_on_test(learner, test)
        model_cache.save({"predicted.json": prediction.to_dict("records")})
        mlflow.log_artifact(os.path.join(
            model_cache.cache_dir, "predicted.json"), "")


if __name__ == "__main__":
    learning()
