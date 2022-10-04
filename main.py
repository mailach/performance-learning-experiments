import click
import yaml


import logging
from rich.logging import RichHandler

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

from utils.runs import get_run_if_exists
from utils.exceptions import handle_exception
from utils.caching import CacheHandler


logging.basicConfig(
    level=logging.INFO,
    format="MAIN    %(message)s",
    handlers=[RichHandler()],
)


def _load_and_cache(
    experiment_name: str, params: dict[str, str] = None, run_id: str = None
):
    if not run_id:
        run_id = get_run_if_exists(experiment_name, params)

    cache = CacheHandler(run_id)

    return cache.cache_dir if cache.existing_cache else download_artifacts(run_id=run_id, dst_path=cache.cache_dir)


def _load_system(params: dict[str, str], param_file: str) -> str:
    logging.info("System loading...")

    if not params["local"]["run_id"]:
        mlflow.run(
            ".",
            entry_point="system_loading",
            parameters={"param_file": param_file},
            experiment_name="system",
        )
        mlflow.log_params(params["parameter"])
    else:
        dir = _load_and_cache("system", run_id=params["local"]["run_id"])
        return params["local"]["run_id"]


@ click.command()
@ click.option("--param_file", default="run.yaml")
def workflow(param_file: str):
    logging.info("Loading parameters...")
    with open(param_file, "r") as f:
        parameters = yaml.safe_load(f)

    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    logging.info("Start execution of workflow")
    with mlflow.start_run() as active_run:

        system_run_id = _load_system(parameters["system"], param_file)


if __name__ == "__main__":
    workflow()
