"""
Takes local data and saves it as artifacts
"""

import logging
import mlflow
import click

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
@click.option("--data_dir")
@click.option("--system", default=None)
@click.option("--domain", default=None)
@click.option("--year", default=None)
@click.option("--authors", default=None)
@click.option("--target", default=None)
@click.option("--n_features_bin", default=0)
@click.option("--shema", default="shema2015")
def load_system(
    data_dir: str,
    system: str,
    domain: str,
    year: str,
    authors: str,
    target: str,
    n_features_bin: int,
    shema: str,
) -> None:
    # parameters to be tracked by mlflow
    params = {
        param: val
        for param, val in locals().items()
        if param not in ["shema", "data_dir"]
    }

    with mlflow.start_run() as run:
        cache = CacheHandler(run.info.run_id)

        logging.info("Load feature model.")
        fm = Fm_handler(data_dir, shema)

        logging.info("Transform xml measurements to one-hot-encoded")
        mh = Measurement_handler(data_dir, fm.features)

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
        mlflow.log_artifacts(cache.cache_dir, "artifacts")
        mlflow.log_params(params)


if __name__ == "__main__":
    load_system()
