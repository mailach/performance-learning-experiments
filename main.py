import click
import os
import tempfile
import logging

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils import mlflow_tags
from mlflow.entities import Run, Experiment
from mlflow.artifacts import download_artifacts
from mlflow.utils.logging_utils import eprint


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s Sampling:  %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
)


def _find_or_create_exp_id(entrypoint, client):
    exp_id = [
        exp.experiment_id for exp in client.list_experiments() if exp.name == entrypoint
    ]
    return exp_id[0] if exp_id else client.create_experiment(entrypoint)


def _same_run(run: Run, params: dict[str:any]):

    same_parameter = [
        True if run.data.params.get(param) == str(params[param]) else False
        for param in params.keys()
    ]

    return True if all(same_parameter) else False


def _get_run_if_exists(
    entrypoint: str, parameters: dict[str, any], git_commit: str, client: MlflowClient
):
    exp_id = _find_or_create_exp_id(entrypoint, client)
    runs = client.list_run_infos(exp_id)
    for run in runs:
        return (
            run.run_id
            if _same_run(client.get_run(run.run_id), parameters)
            and run.status != "FAILED"
            else False
        )


def _load_or_run(
    entrypoint: str, params: dict[str:any], git_commit: str, client: MlflowClient
):
    run_id = _get_run_if_exists(entrypoint, params, git_commit, client)

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
@click.option("--test", default=" ")
@click.option("--data", default=None)
def workflow(test: str, data: str):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:

        logging.info("Generate cache directory")

        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)

        client = MlflowClient()

        cache = _load_or_run(
            "sampling",
            {
                "system_path": "data/Apache/2012",
                "n": 12,
                "method": "true_random",
            },
            git_commit,
            client,
        )
        logging.info(f"Sampling is cached in {cache}.")


if __name__ == "__main__":
    workflow()
