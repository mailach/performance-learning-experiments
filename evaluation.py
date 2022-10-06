import click


import mlflow
import mlflow.sklearn
from mlflow.artifacts import download_artifacts

import pandas as pd
from sklearn import tree

import logging
from rich.logging import RichHandler


from utils.caching import CacheHandler
from utils.runs import get_all_runs, update_metrics, update_params

from learning.models import cart, LearnerFactory, Learner
from learning.metrics import prediction_fault_rate


logging.basicConfig(
    level=logging.INFO,
    format="EVALUATION    %(message)s",
    handlers=[RichHandler()],
)


def _load_learner(run_id: str) -> Learner:
    run = mlflow.get_run(run_id)
    cache = CacheHandler(run_id)
    if not cache.existing_cache:
        logging.info(f"Load artifacts for run {run_id} to cache")
        download_artifacts(run_id=run_id, dst_path=cache.cache_dir)

    learner = LearnerFactory(run.data.params["method"])
    learner.load(cache.cache_dir)
    return learner


def _evaluate(run_id: str, test: pd.DataFrame) -> None:
    learner = _load_learner(run_id)
    fr = prediction_fault_rate(
        test["measured_value"], learner.predict(
            test.drop("measured_value", axis=1))
    )
    update_metrics(run_id, fr)
    return fr


@click.command(
    help="Evaluate one or multiple models from prior learning runs.",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option("--workflow_id")
@click.option("--sampling_run_id")
def evaluate(workflow_id: str, sampling_run_id: str):
    sampling_cache = CacheHandler(sampling_run_id)
    test_data = pd.DataFrame(
        sampling_cache.retrieve("remaining_configurations.json"))
    learner_runs = get_all_runs("learning", {"workflow_id": workflow_id})
    logging.info(f"Evaluate models from runs {learner_runs}")

    metrics = [{"id": run_id, "metrics": _evaluate(run_id, test_data)}
               for run_id in learner_runs]
    best = sorted(
        metrics, key=lambda d: d['metrics']['mean_fault_rate'])[0]

    logging.info(f"Evaluated trained models. Best run: {best}")

    update_metrics(workflow_id, best["metrics"])
    update_params(workflow_id, {"best_learning_run": best["id"]})


if __name__ == "__main__":
    evaluate()
