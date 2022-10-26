import logging

import click
import mlflow


from rich.logging import RichHandler

from utils.caching import CacheHandler
from utils.runs import update_metrics, update_params

from pim.learning.metrics import prediction_fault_rate


logging.basicConfig(
    level=logging.INFO,
    format="EVALUATION    %(message)s",
    handlers=[RichHandler()],
)


def _evaluate(run_id: str) -> None:
    cache = CacheHandler(run_id, new_run=False)
    pred = cache.retrieve("predicted.tsv")
    fault_rate = prediction_fault_rate(pred["measured"], pred["predicted"])
    update_metrics(run_id, fault_rate)
    return fault_rate


@click.command(
    help="Evaluate one or multiple models from prior learning runs.",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option("--workflow_id")
def evaluate(workflow_id: str = ""):
    """
    Evaluation of learning runs

    Parameters
    ----------
    workflow_id : str
        run that corresponds to parent run of runs that should be evaluated
    """
    learner_runs = mlflow.search_runs(
        experiment_names=["learning"],
        filter_string=f"tags.mlflow.parentRunId='{workflow_id}' AND attribute.status = 'FINISHED'",
    )
    if learner_runs.empty:
        logging.warning("No runs for evaluation found. Did you use cached results?")
    else:
        logging.info("Evaluate prediction from runs %s", learner_runs["run_id"])

        metrics = [
            {"id": run_id, "metrics": _evaluate(run_id)}
            for run_id in learner_runs["run_id"]
        ]
        best = sorted(metrics, key=lambda d: d["metrics"]["mean_fault_rate"])[0]

        logging.info("Evaluated trained models. Best run: %s", best)

        update_metrics(workflow_id, best["metrics"])
        update_params(workflow_id, {"best_learning_run": best["id"]})


if __name__ == "__main__":
    evaluate()
