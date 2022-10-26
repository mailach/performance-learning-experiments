import click
import mlflow

import pandas as pd

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
    pred = cache.retrieve("predicted.tsv")
    fr = prediction_fault_rate(pred["measured"], pred["predicted"])
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
def evaluate(workflow_id: str):

    learner_runs = mlflow.search_runs(
        experiment_names=["learning"],
        filter_string=f"tags.mlflow.parentRunId='{workflow_id}' AND attribute.status = 'FINISHED'",
    )
    if learner_runs.empty:
        logging.warning("No runs for evaluation found. Did you use cached results?")
    else:
        logging.info(f"Evaluate prediction from runs {learner_runs['run_id']}")

        metrics = [
            {"id": run_id, "metrics": _evaluate(run_id)}
            for run_id in learner_runs["run_id"]
        ]
        best = sorted(metrics, key=lambda d: d["metrics"]["mean_fault_rate"])[0]

        logging.info(f"Evaluated trained models. Best run: {best}")

        update_metrics(workflow_id, best["metrics"])
        update_params(workflow_id, {"best_learning_run": best["id"]})


if __name__ == "__main__":
    evaluate()
