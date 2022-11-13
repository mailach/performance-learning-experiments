from runs import update_metrics
from caching import CacheHandler
from rich.logging import RichHandler
import logging
from sklearn.metrics import mean_squared_error
import click
import pandas as pd


def prediction_fault_rate(y: pd.Series, y_hat: pd.Series):
    """
    Calculates prediction fault rate.

    Parameters
    ----------
    y : pd.DataFrame
        measured value
    y_hat: pd.DataFrame
        predicted value
    """
    if any(y == 0):
        raise Exception("True value can not be zero.")

    fault_rate = abs(y - y_hat) / y

    return {
        "mean_fault_rate": fault_rate.mean(),
        "median_fault_rate": fault_rate.median(),
        "sd_fault_rate": fault_rate.std(),
    }


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
