import logging

import yaml
import click
import mlflow
from rich.logging import RichHandler


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
    #run_id = get_run_if_exists(entrypoint, params) if use_cache else False

    # if run_id:
    #     logging.info("Use existing run %s for entrypoint %s",
    #                  run_id, entrypoint)
    #     CacheHandler(run_id, new_run=False)
    # else:
    logging.info("Start new run for entrypoint %s", entrypoint)
    run_id = mlflow.run(
        ".",
        entry_point=entrypoint,
        parameters=params,
        experiment_name=entrypoint
    ).run_id

    return run_id


def _log_params(prefix: str, params: dict[str, str]) -> None:
    params = {prefix + "_" + key: value for key, value in params.items()}
    mlflow.log_params(params)


def _load_system(params: dict[str, str], param_file: str) -> str:
    logging.info("Load system %s", params["parameter"]["system"])
    if not params["local"]["run_id"]:
        return _run_or_load("systems", {"param_file": param_file}, use_cache=False)
    else:
        CacheHandler(params["local"]["run_id"], new_run=False)
        return params["local"]["run_id"]


@click.command()
@click.option("--param_file", default="run.yaml")
def workflow(param_file: str = "run.yaml"):
    """
    Function that executes a multistep workflow.

    Parameters
    ----------
    param_file : str
        file that contains parameters.
    """
    logging.info("Loading parameters...")

    import os

    logging.error(os.environ.get("MLFLOW_TRACKING_URI"))
    with open(param_file, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    logging.info("Start execution of workflow as new mlflow run...")
    with mlflow.start_run() as active_run:

        learning_params = {
            "method": params["learning"]["method"],
            "nfp": params["learning"]["nfp"],
        }
        learning_params.update(
            {
                k: v
                for k, v in params["learning"][learning_params["method"]].items()
                if v
            }
        )

        _log_params("system", params["system"]["parameter"])
        _log_params("sampling", params["sampling"])
        _log_params("learning", learning_params)

        system_run_id = _load_system(params["system"], param_file)

        params["sampling"]["system_run_id"] = system_run_id
        sampling_run_id = _run_or_load("sampling", params["sampling"])

        learning_params["sampling_run_id"] = sampling_run_id
        learning_run_id = _run_or_load(
            "learning", learning_params, use_cache=False)

        evaluation_run_id = _run_or_load(
            "evaluation",
            {"workflow_id": active_run.info.run_id},
        )

        mlflow.log_params(
            {
                "system_run_id": system_run_id,
                "sampling_run_id": sampling_run_id,
                "learning_run_id": learning_run_id,
                "evaluation_run_id": evaluation_run_id,
            }
        )


if __name__ == "__main__":
    workflow()
