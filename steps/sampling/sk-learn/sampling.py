import sys
import pandas as pd
from rich.logging import RichHandler
import logging
import mlflow
import click
from sklearn.model_selection import train_test_split
from caching import CacheHandler


mlflow.set_experiment("sampling")
logging.basicConfig(
    filename="logs.txt",
    level=logging.INFO,
    format="SAMPLING    %(message)s",
    # handlers=[RichHandler()],
)


def _split_dataset_by_samples(data, samples):
    data = data.merge(samples, on=list(samples.columns), how="left", indicator=True)
    train = data[data["_merge"] == "both"].drop("_merge", axis=1)
    test = data[data["_merge"] == "left_only"].drop("_merge", axis=1)
    return train, test


def true_random_sampling(n: int, all_configs: pd.DataFrame):

    train, test = train_test_split(all_configs, train_size=n)

    return train, test


@click.command(help="Sample from feature model or list of configurations.")
@click.option("--system_run_id", default="")
@click.option("--method", default=None)
@click.option("--n", default=10, type=int)
def sample(
    method: str,
    n: int = 10,
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
        data = system_cache.retrieve("measurements.tsv")

        logging.info("Sampling using '%s'.", method)
        logging.warning(
            "Only use this method when all valid configurations are available."
        )
        if method == "random":
            train, test = true_random_sampling(int(n), all_configs=data)
        else:
            logging.error("Method not found, exiting...")
            sys.exit(1)

        logging.info("Save sampled configurations to cache")
        sampling_cache.save(
            {
                "train.tsv": train,
                "test.tsv": test,
            }
        )
        logging.info("Sampling cache dir: %s", sampling_cache.cache_dir)
        mlflow.log_artifacts(sampling_cache.cache_dir, "")
        mlflow.log_artifact("logs.txt", "")


if __name__ == "__main__":
    # pylint: disable-next=no-value-for-parameter
    sample()
