from runs import update_metrics
from caching import CacheHandler
import logging
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import click
import pandas as pd
import mlflow


logging.basicConfig(
    filename="logs.txt",
    level=logging.INFO,
    format="EVALUATION    %(message)s",
)


@click.command(
    help="Evaluate one or multiple models from prior learning runs.",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option("--learning_run_id")
def evaluate(learning_run_id: str = ""):
    """
    Evaluation of learning runs

    Parameters
    ----------
    learning_run_id : str
        run that corresponds to learning run that should be evaluated
    """
    cache = CacheHandler(learning_run_id, new_run=False)
    pred = cache.retrieve("predicted.tsv")
    mae = mean_absolute_error(pred["measured"], pred["predicted"])
    mape = mean_absolute_percentage_error(pred["measured"], pred["predicted"])
    update_metrics(learning_run_id, {"mape": mape, "mre": mape / 100, "mae": mae})
    # mlflow.log_artifact("logs.txt", "")


if __name__ == "__main__":
    evaluate()
