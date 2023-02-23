import logging
import click

import mlflow
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from caching import CacheHandler


def activate_logging(logs_to_artifact):
    with open("logs.txt", "w", encoding="utf-8"):
        pass
    if logs_to_artifact:
        return logging.basicConfig(
            filename="logs.txt",
            level=logging.INFO,
            format="EVALUATION    %(message)s",
        )
    return logging.basicConfig(
        level=logging.INFO,
        format="EVALUATION    %(message)s",
    )


def update_metrics(run_id: str, metric: dict[str, any]):
    """
    Updates the metrics of an already existing run.

    Parameters
    ----------
    run_id : str
        run to update
    metric : dict[str, any]
        metrics to update
    """
    with mlflow.start_run(run_id):
        mlflow.log_metrics(metric)


@click.command(
    help="Evaluate one or multiple models from prior learning runs.",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option("--learning_run_id")
@click.option("--logs_to_artifacts", type=bool, default=True)
def evaluate(
    learning_run_id: str = "",
    logs_to_artifacts: bool = False,
):
    """
    Evaluation of learning runs

    Parameters
    ----------
    learning_run_id : str
        run that corresponds to learning run that should be evaluated
    """
    activate_logging(logs_to_artifacts)
    logging.info("Start evaluation...")
    cache = CacheHandler(learning_run_id, new_run=False)
    pred = cache.retrieve("predicted.tsv")
    mae = mean_absolute_error(pred["measured"], pred["predicted"])
    mape = mean_absolute_percentage_error(pred["measured"], pred["predicted"])

    logging.info("Update learning runs...")
    update_metrics(learning_run_id, {"mape": mape, "mre": mape / 100, "mae": mae})
    if logs_to_artifacts:
        mlflow.log_artifact("logs.txt", "")


if __name__ == "__main__":
    evaluate()
