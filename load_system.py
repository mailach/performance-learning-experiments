"""
Takes local data and saves it as artifacts
"""

import logging
import os
import mlflow
import click
import yaml

from rich.logging import RichHandler

from pim.feature_models.modeling import FeatureModel
from pim.feature_models.transformations import Measurements
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
    param_file: str = "run.yaml",
) -> None:
    """
    Loads, validates, and transforms data of system.

    Parameters
    ----------
    param_file : str
        file that contains parameters.
    """

    with open(param_file, "r", encoding="utf-8") as file:
        run_file = yaml.safe_load(file)
        params = run_file["system"]["parameter"]
        local = run_file["system"]["local"]

    with mlflow.start_run() as run:
        cache = CacheHandler(run.info.run_id)

        logging.info("Load feature model.")
        feature_model = FeatureModel(os.path.join(local["data_dir"], "fm.xml"))

        logging.info("Transform xml measurements to one-hot-encoded")
        measurements = Measurements(
            os.path.join(local["data_dir"], "all_measurements.xml"),
            feature_model.binary,
            feature_model.numeric,
        )

        mlflow.log_params(params)
        cache.save(
            {
                "fm.xml": feature_model.xml,
                "fm.dimacs": feature_model.dimacs,
                "features.json": feature_model.get_features(),
                "measurements.xml": measurements.xml,
                "measurements.tsv": measurements.df,
            }
        )

        logging.info("Log artifacts and parameters to MLflow")
        mlflow.log_artifacts(cache.cache_dir, "")
        mlflow.log_params(params)


if __name__ == "__main__":
    load_system()
