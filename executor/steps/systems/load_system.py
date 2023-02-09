"""
Takes local data and saves it as artifacts
"""
import logging
import os
import sys
import mlflow
import click
import yaml

from rich.logging import RichHandler

from modeling import FeatureModel
from transformations import Measurements
from caching import CacheHandler

logging.basicConfig(
    level=logging.INFO,
    format="LOAD SYSTEM    %(message)s",
    handlers=[RichHandler()],
)


mlflow.set_experiment("systems")


def _check_mandatory_files(data_dir: str):
    files = os.listdir(data_dir)
    mandatory_files = ["fm.xml", "meta.yaml", "measurements.xml"]
    missing = [f for f in mandatory_files if f not in files]
    if len(missing):
        logging.error("You need to provide %s. \nExiting...", " and ".join(missing))
        sys.exit(1)


def _check_measurement_files(data_dir: str):
    files = os.listdir(data_dir)
    missing = [f for f in ["measurements.xml", "measurements.tsv"] if f not in files]
    if len(missing) > 1:
        logging.error(
            "You need to provide either %s. \nExiting...", " or ".join(missing)
        )
        sys.exit(1)


def _check_dir_content(data_dir: str):
    if not os.path.exists(data_dir):
        logging.error("Path %s does not exist. \nExiting...", data_dir)
        sys.exit(1)
    _check_mandatory_files(data_dir)
    # _check_measurement_files(data_dir)


@click.command(help="Imports feature model and measurement data.")
@click.option("--data_dir")
@click.option("--logs_to_artifact", type=bool, default=False)
def load_system(data_dir: str, logs_to_artifact: bool = False):
    """
    Loads, validates, and transforms data of system.

    Parameters
    ----------
    data_dir : str
        Directory that contains system data. The following files are mandatory:
        - meta.yaml
        - fm.xml
        - measurements.xml or measurements.tsv
    """

    _check_dir_content(data_dir)
    with open(os.path.join(data_dir, "meta.yaml"), "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    logging.info("Load feature model...")
    feature_model = FeatureModel(os.path.join(data_dir, "fm.xml"))

    logging.info("Load and transform measurements...")
    measurements = Measurements(
        os.path.join(data_dir, "measurements.xml"),
        feature_model.binary,
        feature_model.numeric,
    )

    with mlflow.start_run() as run:
        logging.info("Start mlflow run for systems...")
        cache = CacheHandler(run.info.run_id)
        cache.save(
            {
                "fm.xml": feature_model.xml,
                "fm.dimacs": feature_model.dimacs,
                "features.json": feature_model.get_features(),
                "measurements.tsv": measurements.df,
                "meta.json": params,
            }
        )

        logging.info("Log artifacts and parameters to MLflow")
        mlflow.log_artifacts(cache.cache_dir, "")
        mlflow.log_params(params)


if __name__ == "__main__":
    # pylint: disable-next=no-value-for-parameter
    load_system()
