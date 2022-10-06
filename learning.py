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

def _predict_on_test(learner: Learner, test_X: pd.DataFrame, test_Y: pd.Series):
    pred = pd.Series(learner.predict(test_X))
    pred.name = "predicted"
    return pd.concat([pred, test_Y], axis=1)


def _load_data(data_file: str, cache: CacheHandler):
    data = pd.DataFrame(cache.retrieve(data_file))
    Y = data["measured_value"]
    X = data.drop("measured_value", axis=1)
    return X, Y


@ click.command(
    help="Learn from sampled configurations",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),


)
@ click.option("--sampling_run_id")
@ click.option("--method")
@ click.option("--workflow_id")
@ click.option("--min_samples_split", type=int, default=2)
@ click.option("--min_samples_leaf", type=int, default=1)
def learning(
    sampling_run_id: str, workflow_id: str, method: str, min_samples_split: int, min_samples_leaf: int
):

    logging.info("Start learning from sampled configurations.")

    sampling_cache = CacheHandler(sampling_run_id)

    train_X, train_Y = _load_data(
        "sampled_configurations.json", sampling_cache)
    test_X, test_Y = _load_data(
        "remaining_configurations.json", sampling_cache)

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
        prediction = _predict_on_test(learner, test_X, test_Y)
        model_cache.save({"predicted.json": prediction.to_dict("records")})
        mlflow.log_artifact(os.path.join(
            model_cache.cache_dir, "predicted.json"), "")


if __name__ == "__main__":
    learning()
