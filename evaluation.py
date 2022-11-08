from pim.learning.metrics import prediction_fault_rate
from utils.runs import update_metrics, update_params
from utils.caching import CacheHandler
from rich.logging import RichHandler
import logging
from sklearn.metrics import mean_squared_error


import click
import mlflow


logging.basicConfig(
    level=logging.INFO,
    format="EVALUATION    %(message)s",
    handlers=[RichHandler()],
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
    fault_rate = prediction_fault_rate(pred["measured"], pred["predicted"])
    mse = mean_squared_error(pred["measured"], pred["predicted"])
    update_metrics(learning_run_id, fault_rate)
    update_metrics(learning_run_id, {"test_mse": mse})


if __name__ == "__main__":
    evaluate()
