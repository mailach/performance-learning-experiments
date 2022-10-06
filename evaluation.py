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

from learning.metrics import prediction_fault_rate


logging.basicConfig(
    level=logging.INFO,
    format="EVALUATION    %(message)s",
    handlers=[RichHandler()],
)


def _evaluate(run_id: str) -> None:
    cache = CacheHandler(run_id)
    pred = pd.DataFrame(cache.retrieve("predicted.json"))
    fr = prediction_fault_rate(pred["measured_value"], pred["predicted"])
    update_metrics(run_id, fr)
    return fr


@ click.command(
    help="Evaluate one or multiple models from prior learning runs.",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),


)
@ click.option("--workflow_id")
def evaluate(workflow_id: str):

    learner_runs = get_all_runs("learning", {"workflow_id": workflow_id})
    logging.info(f"Evaluate prediction from runs {learner_runs}")

    metrics = [{"id": run_id, "metrics": _evaluate(run_id)}
               for run_id in learner_runs]
    best = sorted(
        metrics, key=lambda d: d['metrics']['mean_fault_rate'])[0]

    logging.info(f"Evaluated trained models. Best run: {best}")

    update_metrics(workflow_id, best["metrics"])
    update_params(workflow_id, {"best_learning_run": best["id"]})


if __name__ == "__main__":
    evaluate()
