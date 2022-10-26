import logging
import random
import mlflow
import click

import logging
from rich.logging import RichHandler

from typing import Sequence
import pandas as pd

from utils.caching import CacheHandler
from sampling.binary import SamplerFactory


mlflow.set_experiment("sampling")
logging.basicConfig(
    level=logging.INFO,
    format="SAMPLING    %(message)s",
    handlers=[RichHandler()],
)


@click.command(help="Sample from feature model or list of configurations.")
@click.option("--system_run_id", default="")
@click.option("--method", default="true_random")
@click.option("--n", default=10, type=int)
def sample(n: int, method: str, system_run_id: str):

    logging.info("Start sampling from configuration space.")

    sampler = SamplerFactory(method)

    with mlflow.start_run() as run:

        sampling_cache = CacheHandler(run.info.run_id)
        system_cache = CacheHandler(system_run_id, new_run=False)

        if method == "true_random":
            logging.info("Sampling using 'true random'.")
            logging.warning(
                "Only use this method when all valid configurations are available."
            )
            configurations = system_cache.retrieve("measurements.tsv")
            sampled_configs, remaining_configs = sampler.sample(
                int(n), all_configs=configurations
            )

        else:
            logging.error("Sampling method not implemented yet.")
            raise NotImplementedError

        logging.info(f"Save sampled configurations to cache")
        sampling_cache.save(
            {
                "train.tsv": sampled_configs,
                "test.tsv": remaining_configs,
            }
        )

        mlflow.log_artifact(sampling_cache.cache_dir, "")


if __name__ == "__main__":
    sample()
