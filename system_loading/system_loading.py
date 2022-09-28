"""
Takes local data and saves it as artifacts
"""
import os
import yaml
import logging
import mlflow
import click
from utils.fm import fm_xml_to_dimacs
from utils.data import xml_measurements_to_onehot

mlflow.set_experiment("systems")





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


@click.command(help="Takes local csv and saves it for later use as artifact")
@click.option("--path", default=None)
@click.option("--system", default=None)
@click.option("--year", default=None)
def load_system(path: str, system: str, year: int) -> None:

    data_dir = os.path.join(path, system, year)
    logging.info("Transform featuremodel in xml to dimacs (cnf)")
    fm_xml_to_dimacs(data_dir, "shema2015")

    logging.info("Transform xml measurements to one-hot-encoded")
    xml_measurements_to_onehot(data_dir)

    logging.info("Loading metadata...")
    with mlflow.start_run():
        try:
            meta_f = os.path.join(path, system, "data.yaml")
            with open(meta_f, "r") as f:
                metadata = yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Can't load metadata from file: {meta_f}")
            raise e

        logging.info("Log metadata as parameters...")
        _log_metadata(metadata, int(year))

        logging.info(f"Uploading data file to {os.environ['MLFLOW_TRACKING_URI']}")
        mlflow.log_artifact(os.path.join(path, system, year), "data")


if __name__ == "__main__":
    load_system()
