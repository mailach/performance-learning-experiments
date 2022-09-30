import click
import os
import tempfile

import logging
from rich.logging import RichHandler

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils import mlflow_tags
from mlflow.entities import Run, Experiment
from mlflow.artifacts import download_artifacts

from utils.runs import get_run_if_exists
from utils.exceptions import handle_exception


logging.basicConfig(
    level=logging.INFO,
    format="MAIN    %(message)s",
    handlers=[RichHandler()],
)


@handle_exception("Not able to load or run entrypoint.")
def _load_or_run(
    entrypoint: str, params: dict[str:any], git_commit: str, client: MlflowClient
):
    run_id = get_run_if_exists(entrypoint, params, git_commit, client)

    if run_id:
        logging.info(
            f"Found existing run for entrypoint {entrypoint}. Caching artifacts..."
        )
        return download_artifacts(run_id=run_id)
    else:
        logging.info(f"Start new run for entrypoint {entrypoint}.")
        cache = tempfile.mkdtemp()
        mlflow.run(
            ".",
            "sampling",
            parameters=params,
            experiment_name=entrypoint,
        )


@click.command()
@click.option("--sampling_method", default="true_random")
@click.option("--sampling_n", default=5, type=int)
def workflow(sampling_n: int, sampling_method: str):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("runs")
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    logging.info("Start execution of workflow")
    with mlflow.start_run() as active_run:

        logging.info("Generate cache directory")

        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)

        client = MlflowClient()
        cache_dir = "data/Apache/2012"  # tempfile

        cache = _load_or_run(
            "sampling",
            {
                "data_dir": cache_dir,
                "n": sampling_n,
                "method": sampling_method,
            },
            git_commit,
            client,
        )
        logging.info(f"Sampling is cached in {cache}.")


if __name__ == "__main__":
    workflow()
