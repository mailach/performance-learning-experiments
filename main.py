import click
import yaml


import logging
from rich.logging import RichHandler

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

from utils.runs import get_run_if_exists
from utils.exceptions import handle_exception


logging.basicConfig(
    level=logging.INFO,
    format="MAIN    %(message)s",
    handlers=[RichHandler()],
)


# @handle_exception("Not able to load or run entrypoint.")
# def _load_or_run(
#     entrypoint: str, params: dict[str:any], git_commit: str, client: MlflowClient
# ):
#     run_id = get_run_if_exists(entrypoint, params)

#     if run_id:
#         logging.info(
#             f"Found existing run for entrypoint {entrypoint}. Caching artifacts..."
#         )
#         return download_artifacts(run_id=run_id)
#     else:
#         logging.info(f"Start new run for entrypoint {entrypoint}.")
#         cache = tempfile.mkdtemp()
#         mlflow.run(
#             ".",
#             "sampling",
#             parameters=params,
#             experiment_name=entrypoint,
#         )


def _load_and_cache(
    experiment_name: str, params: dict[str, str] = None, run_id: str = None
):
    if not run_id:
        run_id = get_run_if_exists(experiment_name, params)

    return download_artifacts(run_id=run_id)


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
        print("Run exists and cached: " + dir)



@click.command()
@click.option("--param_file", default="run.yaml")
# @click.option("--sampling_method", default="true_random")
# @click.option("--sampling_n", default=5, type=int)
def workflow(param_file: str):
    logging.info("Loading parameters...")
    with open(param_file, "r") as f:
        parameters = yaml.safe_load(f)

    
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    logging.info("Start execution of workflow")
    with mlflow.start_run() as active_run:
        _load_system(parameters["system"], param_file)

        # existing_run = get_run_if_exists("system_loading", params)

        # git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)

        # client = MlflowClient()
        # cache_dir = "data/Apache/2012"  # tempfile

        # cache = _load_or_run(
        #     "sampling",
        #     {
        #         "data_dir": cache_dir,
        #         "n": sampling_n,
        #         "method": sampling_method,
        #     },
        #     git_commit,
        #     client,
        # )
        # logging.info(f"Sampling is cached in {cache}.")


if __name__ == "__main__":
    workflow()
