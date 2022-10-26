"""
Takes local data and saves it as artifacts
"""

import logging
import os
import mlflow
import click
import yaml

import logging
from rich.logging import RichHandler


from feature_models.modeling import FeatureModel
from feature_models.transformations import Measurements
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
        fm = FeatureModel(os.path.join(local["data_dir"], "fm.xml"))

        logging.info("Transform xml measurements to one-hot-encoded")
        mh = Measurements(
            os.path.join(local["data_dir"], "all_measurements.xml"),
            fm.binary,
            fm.numeric,
        )

        mlflow.log_params(params)
        cache.save(
            {
                "fm.xml": fm.xml,
                "fm.dimacs": fm.dimacs,
                "features.json": fm.get_features(),
                "measurements.xml": mh.xml,
                "measurements.tsv": mh.df,
            }
        )

        logging.info("Log artifacts and parameters to MLflow")
        mlflow.log_artifacts(cache.cache_dir, "")
        mlflow.log_params(params)


if __name__ == "__main__":
    load_system()
