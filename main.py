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


@click.command()
@click.option("--param_file", default="run.yaml")
# @click.option("--sampling_method", default="true_random")
# @click.option("--sampling_n", default=5, type=int)
def workflow(param_file: str):
    with open(param_file, "r") as f:
        parameters = yaml.safe_load(f)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("runs")
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    logging.info("Start execution of workflow")
    with mlflow.start_run() as active_run:

        logging.info("System loading...")

        system = parameters["system"]
        if not system["run_id"]:
            mlflow.run(
                ".",
                entry_point="system_loading",
                parameters=system,
                experiment_name="system",
            )
        else:
            dir = _load_and_cache("system", run_id=system["run_id"])
        print(dir)

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
