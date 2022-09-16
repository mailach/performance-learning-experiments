import logging
import random
import json
import os
import mlflow
import click
import tempfile

from typing import Sequence


mlflow.set_experiment("test")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s Sampling:  %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
)


def _true_random(configs: Sequence, n: int) -> Sequence:

    if len(configs) < n:
        logging.error(
            f"Desired sample size n={n} is smaller than number of available configurations n={len(configs)}. "
        )
        raise Exception("Valueerror for samplesize.")

    return random.sample(configs, n)


@click.command(help="Sample from feature model or list of configurations.")
@click.option("--n", default=None)
@click.option("--system_path", default=None)
@click.option("--method", default=None)
def sample(n: int, system_path: str, method: str):

    logging.info("Start sampling from configuration space.")

    with mlflow.start_run():
        if method == "true_random":
            logging.info("Sampling using 'true random'.")
            logging.warning(
                "Only use this method when all valid configurations are available."
            )
            with open(os.path.join(system_path, "measurements.json"), "r") as f:
                configurations = json.load(f)
            sampled_configs = _true_random(configurations, int(n))
        else:
            logging.error("Sampling method not implemented yet.")
            raise NotImplementedError

        configs_file = os.path.join(tempfile.mkdtemp(), "sampled_configurations.json")
        with open(configs_file, "w") as f:
            json.dump(sampled_configs, f)

        mlflow.log_param("n_sample", n)
        mlflow.log_artifact(configs_file, "sampled_configurations")


if __name__ == "__main__":
    sample()
