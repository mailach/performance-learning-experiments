import click
import yaml


import logging
from rich.logging import RichHandler

import mlflow
from mlflow.artifacts import download_artifacts

from utils.runs import get_run_if_exists
from utils.caching import CacheHandler


logging.basicConfig(
    level=logging.INFO,
    format="MAIN    %(message)s",
    handlers=[RichHandler()],
)


def _run_or_load(
    entrypoint: str, params: dict[str, str], use_cache: bool = True
) -> str:
    run_id = get_run_if_exists(entrypoint, params) if use_cache else False

    if run_id:
        logging.info(f"Use existing run {run_id} for entrypoint {entrypoint}")
        _load_and_cache(entrypoint, run_id)
    else:
        logging.info(f"Start new run for entrypoint {entrypoint}")
        run_id = mlflow.run(
            ".",
            entry_point=entrypoint,
            parameters=params,
            experiment_name=entrypoint,
        ).run_id

    return run_id


def _load_and_cache(experiment_name: str, run_id: str = None) -> None:

    cache = CacheHandler(run_id)
    if not cache.existing_cache:
        logging.info(f"Load artifacts for run {run_id} to cache")
        download_artifacts(run_id=run_id, dst_path=cache.cache_dir)
    else:
        logging.info(f"Use existing cache for run {run_id}")


def _log_params(prefix: str, params: dict[str, str]) -> None:
    params = {prefix + "_" + key: value for key, value in params.items()}
    mlflow.log_params(params)


def _load_system(params: dict[str, str], param_file: str) -> str:
    logging.info(f"Load system {params['parameter']['system']}")
    if not params["local"]["run_id"]:
        return _run_or_load("systems", {"param_file": param_file}, use_cache=False)
    else:
        dir = _load_and_cache("system", run_id=params["local"]["run_id"])
        return params["local"]["run_id"]


@click.command()
@click.option("--param_file", default="run.yaml")
def workflow(param_file: str):
    logging.info("Loading parameters...")
    with open(param_file, "r") as f:
        params = yaml.safe_load(f)

    logging.info("Start execution of workflow as new mlflow run...")
    with mlflow.start_run() as active_run:

        learning_params = {"method": params["learning"]["method"]}
        learning_params.update(params["learning"][learning_params["method"]])

        _log_params("system", params["system"]["parameter"])
        _log_params("sampling", params["sampling"])
        _log_params("learning", learning_params)

        system_run_id = _load_system(params["system"], param_file)

        params["sampling"]["system_run_id"] = system_run_id
        sampling_run_id = _run_or_load("sampling", params["sampling"])

        learning_params["sampling_run_id"] = sampling_run_id
        learning_params["workflow_id"] = active_run.info.run_id
        learning_run_id = _run_or_load("learning", learning_params)

        evaluation_run_id = _run_or_load(
            "evaluation",
            {"workflow_id": active_run.info.run_id, "sampling_run_id": sampling_run_id},
        )

        mlflow.log_params(
            {
                "system_run_id": system_run_id,
                "sampling_run_id": sampling_run_id,
                "learning_run_id": learning_run_id,
            }
        )


if __name__ == "__main__":
    workflow()
