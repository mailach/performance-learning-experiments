import logging
import mlflow
import click

from rich.logging import RichHandler
import pandas as pd

from utils.caching import CacheHandler
from pim.sampling.binary import true_random_sampling, binary_sampling, mixed_sampling

mlflow.set_experiment("sampling")
logging.basicConfig(
    level=logging.INFO,
    format="SAMPLING    %(message)s",
    handlers=[RichHandler()],
)


@click.command(help="Sample from feature model or list of configurations.")
@click.option("--system_run_id", default="")
@click.option("--binary_method", default=None)
@click.option("--numeric_method", default=None)
@click.option("--true_random", type=bool, default=None)
@click.option("--n", default=10, type=int)
def sample(
    n: int = 10,
    binary_method: str = None,
    numeric_method: str = None,
    true_random: bool = False,
    system_run_id: str = "",
):
    """
    Samples valid configurations from a variability model.

    Parameters
    ----------
    n: int
        number of samples
    method: str
        method for sampling
    system_run_id : str
        run of system loading
    """

    logging.info("Start sampling from configuration space.")

    with mlflow.start_run() as run:

        sampling_cache = CacheHandler(run.info.run_id)
        system_cache = CacheHandler(system_run_id, new_run=False)
        vm = system_cache.retrieve("fm.xml")
        data = system_cache.retrieve("measurements.tsv")

        if numeric_method == "None":
            numeric_method = None
        if true_random:
            logging.info("Sampling using 'true random'.")
            logging.warning(
                "Only use this method when all valid configurations are available."
            )
            configurations = system_cache.retrieve("measurements.tsv")
            train, test = true_random_sampling(int(n), all_configs=configurations)

        elif binary_method and numeric_method:
            samples = pd.DataFrame(mixed_sampling(binary_method, numeric_method, vm))
            data = data.merge(
                samples, on=list(samples.columns), how="left", indicator=True
            )
            train = data[data["_merge"] == "both"].drop("_merge", axis=1)
            test = data[data["_merge"] == "left_only"].drop("_merge", axis=1)

        elif binary_method:
            samples = pd.DataFrame(binary_sampling(binary_method, vm=vm))
            data = data.merge(
                samples, on=list(samples.columns), how="left", indicator=True
            )
            train = data[data["_merge"] == "both"].drop("_merge", axis=1)
            test = data[data["_merge"] == "left_only"].drop("_merge", axis=1)

        else:
            logging.error("Sampling method not implemented yet.")
            raise NotImplementedError

        logging.info("Save sampled configurations to cache")
        sampling_cache.save(
            {
                "train.tsv": train,
                "test.tsv": test,
            }
        )

        mlflow.log_artifact(sampling_cache.cache_dir, "")


if __name__ == "__main__":
    sample()
