import logging
import random
import json
import os
import mlflow
import click

import logging
from rich.logging import RichHandler

from typing import Sequence

from utils.caching import CacheHandler


mlflow.set_experiment("sampling")
logging.basicConfig(
    level=logging.INFO,
    format="SAMPLING    %(message)s",
    handlers=[RichHandler()],
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
@click.option("--system_run_id", default="")
@click.option("--method", default="true_random")
@click.option("--n", default=10, type=int)
def sample(n: int, method: str, system_run_id: str):

    logging.info("Start sampling from configuration space.")

    with mlflow.start_run() as run:

        sampling_cache = CacheHandler(run.info.run_id)
        system_cache = CacheHandler(system_run_id)
        logging.info(
            f"Initialized cache for system at {system_cache.cache_dir}")
        logging.info(
            f"Initialized cache for sampling at {sampling_cache.cache_dir}")

        if method == "true_random":
            logging.info("Sampling using 'true random'.")
            logging.warning(
                "Only use this method when all valid configurations are available."
            )
            configurations = system_cache.retrieve("measurements_oh.json")
            sampled_configs, remaining_configs = _true_random(
                configurations, int(n))

        else:
            logging.error("Sampling method not implemented yet.")
            raise NotImplementedError

        # Generate cache dir and save sampled configurations
        logging.info(f"Save sampled configurations to cache")
        sampling_cache.save(
            {
                "sampled_configurations.json": sampled_configs,
                "remaining_configurations.json": remaining_configs,
            }
        )

        # log cache as parameter
        mlflow.log_artifact(sampling_cache.cache_dir, "artficacts")
        # mlflow.log_artifact(remain_configs_file, "remaining_configurations")


if __name__ == "__main__":
    sample()
