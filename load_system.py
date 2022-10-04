"""
Takes local data and saves it as artifacts
"""

import logging
import mlflow
import click
import yaml

import logging
from rich.logging import RichHandler


from feature_models.modeling import Fm_handler
from feature_models.transformations import Measurement_handler
from utils.caching import CacheHandler

logging.basicConfig(
    level=logging.INFO,
    format="LOAD SYSTEM    %(message)s",
    handlers=[RichHandler()],
)


@click.command(
    help="Imports feature model data into standard format and saves it as artifacts to MLFlow server."
)
@click.option("--param_file", default="run.yaml")
# @click.option("--data_dir")
# @click.option("--system", default=None)
# @click.option("--domain", default=None)
# @click.option("--year", default=None)
# @click.option("--authors", default=None)
# @click.option("--target", default=None)
# @click.option("--n_features_bin", default=0)
# @click.option("--shema", default="shema2015")
# def load_system(
#     data_dir: str,
#     system: str,
#     domain: str,
#     year: str,
#     authors: str,
#     target: str,
#     n_features_bin: int,
#     shema: str,
# ) -> None:
# parameters to be tracked by mlflow
def load_system(
    param_file: str,
) -> None:

    with open(param_file, "r") as f:
        run_file = yaml.safe_load(f)
        params = run_file["system"]["parameter"]
        local = run_file["system"]["local"]

    with mlflow.start_run() as run:
        cache = CacheHandler(run.info.run_id)

        logging.info("Load feature model.")
        fm = Fm_handler(local["data_dir"], local["shema"])

        logging.info("Transform xml measurements to one-hot-encoded")
        mh = Measurement_handler(local["data_dir"], fm.features)

        mlflow.log_params(params)
        cache.save(
            {
                "fm.xml": fm.xml,
                "fm.dimacs": fm.dimacs,
                "features.json": fm.features,
                "measurements.xml": mh.xml,
                "measurements_oh.json": mh.one_hot,
            }
        )

        logging.info("Log artifacts and parameters to MLflow")
        mlflow.log_artifacts(cache.cache_dir, "")
        mlflow.log_params(params)


if __name__ == "__main__":
    load_system()
