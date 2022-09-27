import logging
import random
import json
import os
import mlflow
import click
import tempfile

from typing import Sequence


mlflow.set_experiment("sampling")


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

    sampled = [configs.pop(random.randrange(len(configs))) for _ in range(n)]

    return sampled, configs


@click.command(help="Sample from feature model or list of configurations.")
@click.option("--n", default=10, type=int)
@click.option("--data_dir", default="")
@click.option("--method", default="true_random")
def sample(n: int, data_dir: str, method: str):

    logging.info("Start sampling from configuration space.")

    with mlflow.start_run():
        if method == "true_random":
            logging.info("Sampling using 'true random'.")
            logging.warning(
                "Only use this method when all valid configurations are available."
            )
            with open(os.path.join(data_dir, "measurements.json"), "r") as f:
                configurations = json.load(f)
            sampled_configs, remaining_configs = _true_random(configurations, int(n))
        else:
            logging.error("Sampling method not implemented yet.")
            raise NotImplementedError

        cache_dir = tempfile.mkdtemp()
        logging.info(f"Save sample to cache at {cache_dir}")
        sampled_configs_file = os.path.join(cache_dir, "sampled_configurations.json")
        with open(sampled_configs_file, "w") as f:
            json.dump(sampled_configs, f)

        logging.info(f"Save remaining configurations to cache at {cache_dir}")
        remain_configs_file = os.path.join(cache_dir, "remaining_configurations.json")
        with open(remain_configs_file, "w") as f:
            json.dump(remaining_configs, f)

        # mlflow.log_param("n_sample", n)
        mlflow.log_artifact(sampled_configs_file, "sampled_configurations")
        mlflow.log_artifact(remain_configs_file, "remaining_configurations")


if __name__ == "__main__":
    sample()
