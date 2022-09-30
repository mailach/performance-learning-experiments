"""
Takes local data and saves it as artifacts
"""
import os
import yaml
import logging
import mlflow
import click
from feature_models.modeling import fm_handler
from feature_models.transformations import xml_measurements_to_onehot
from utils.caching import CacheHandler

# mlflow.set_experiment("systems")


def _log_metadata(metadata: dict, year: int) -> None:
    mlflow.log_param("system", metadata["system"])
    mlflow.log_param("domain", metadata["domain"])

    for m in metadata["measurements"]:
        if year == m["year"]:
            mlflow.log_param("source_year", year)
            mlflow.log_param("source_authors", m["authors"])
            mlflow.log_param("source_link", m["paper"])
            mlflow.log_param("n_configurations", m["configurations"])
            mlflow.log_param("n_features", m["features"])
            mlflow.log_param("all_measured", m["all_measured"])


@click.command(
    help="Imports feature model data into standard format and saves it as artifacts to MLFlow server."
)
@click.option("--data_dir")
# @click.option("--system")
# @click.option("--domain")
# @click.option("--year")
# @click.option("--authors")
# @click.option("--target")
# @click.option("--n_features_bin")
@click.option("--shema")
def load_system(data_dir: str, shema: str) -> None:

    logging.info("Load feature model")

    fm = fm_handler(data_dir, shema)
    print(fm.dimacs)

    logging.info("Transform xml measurements to one-hot-encoded")
    xml_measurements_to_onehot(data_dir)

    # logging.info("Loading metadata...")
    # with mlflow.start_run() as run:
    #     cache = CacheHandler(run.info.run_id)
    #     try:
    #         meta_f = os.path.join(data_dir, "data.yaml")
    #         with open(meta_f, "r") as f:
    #             metadata = yaml.safe_load(f)
    #     except Exception as e:
    #         logging.error(f"Can't load metadata from file: {meta_f}")
    #         raise e

    #     logging.info("Log metadata as parameters...")
    #     _log_metadata(metadata, int(year))

    #     logging.info(f"Uploading data file to {os.environ['MLFLOW_TRACKING_URI']}")
    #     mlflow.log_artifact(os.path.join(path, system, year), "data")


if __name__ == "__main__":
    load_system()
