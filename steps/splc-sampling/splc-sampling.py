#   sampleSize: {type: float, default: None},
#   seed: {type: float, default: None},
#   precision: {type: float, default: None},
#   distinctValuesPerOption: {type: float, default: None},
#   measurements: {type: float, default: None},
#   k: {type: float, default: None},
#   t: {type: float, default: None},
#   optionWeight: {type: float, default: None},
#   numConfigs: {type: float, default: None},
#   numeric_method: {type: float, default: None},


import sys
from caching import CacheHandler
import pandas as pd
from rich.logging import RichHandler
import logging
import mlflow
import click

from splc2py.sampling import Sampler


mlflow.set_experiment("sampling")
logging.basicConfig(
    level=logging.INFO,
    format="SAMPLING    %(message)s",
    handlers=[RichHandler()],
)


def _extract_parameters(params, method, param_dict):
    """returns parameters needed for method"""
    try:
        return (
            {p: params[p] for p in param_dict[method]} if method in param_dict else {}
        )
    except KeyError as k_err:
        logging.error(
            "To run method %s you need to specify parameter %s=",
            method,
            k_err.args[0],
        )
        sys.exit(1)


def _sampling_params(context, bin_method: str, num_method: str = None):
    params = [arg.replace("--") for arg in context.args]
    params = {p.split("=")[0]: p.split("=")[1] for p in params}
    param_dict = {
        "random": ["sampleSize", "seed"],
        "hypersampling": ["precision"],
        "onefactoratatime": ["distinctValuesPerOption"],
        "plackettburman": ["measurements", "level"],
        "kexchange": ["sampleSize", "k"],
        "twise": ["t"],
        "distancebased": ["optionWeight", "numConfigs"],
    }

    samp_params = _extract_parameters(params, bin_method, param_dict)
    samp_params.update(_extract_parameters(params, num_method, param_dict))

    return samp_params


def _split_dataset_by_samples(data, samples):
    data = data.merge(samples, on=list(samples.columns), how="left", indicator=True)
    train = data[data["_merge"] == "both"].drop("_merge", axis=1)
    test = data[data["_merge"] == "left_only"].drop("_merge", axis=1)
    return train, test


@click.command(
    help="Sample using SPLConqueror.",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option("--system_run_id", required=True)
@click.option("--binary_method", required=True)
@click.option("--numeric_method", default=None)
@click.pass_context
# @click.option("--sampleSize", default=None)
# @click.option("--seed", default=None)
# @click.option("--precision", default=None)
# @click.option("--distinctValuesPerOption", default=None)
# @click.option("--measurements", default=None)
# @click.option("--k", default=None)
# @click.option("--t", default=None)
# @click.option("--optionWeight", default=None)
# @click.option("--numConfigs", default=None)
def sample(
    context,
    system_run_id: str,
    binary_method: str,
    # sampleSize: int,
    # seed: int,
    # precision: int,
    # distinctValuesPerOption: int,
    # measurements: int,
    # k: int,
    # t: int,
    # optionWeight: int,
    # numConfigs: int,
    numeric_method: str = None,
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

    logging.info("Start sampling from configuration space using SPLC2py.")
    numeric_method = None if numeric_method == "None" else numeric_method
    system_cache = CacheHandler(system_run_id, new_run=False)
    feature_model = system_cache.retrieve("fm.xml")
    data = system_cache.retrieve("measurements.tsv")

    params = _sampling_params(context, binary_method, numeric_method)
    sampler = Sampler(feature_model, backend="local")

    with mlflow.start_run() as run:
        sampling_cache = CacheHandler(run.info.run_id)

        if not numeric_method:
            samples = pd.DataFrame(
                sampler.sample(binary_method, formatting="dict", params=params)
            )

            train, test = _split_dataset_by_samples(data, samples)

        else:
            samples = pd.DataFrame(
                sampler.sample(
                    binary_method, numeric_method, formatting="dict", params=params
                )
            )
            train, test = _split_dataset_by_samples(data, samples)

        logging.info("Save sampled configurations to cache")
        sampling_cache.save(
            {
                "train.tsv": train,
                "test.tsv": test,
            }
        )
        logging.info("Sampling cache dir: %s", sampling_cache.cache_dir)
        mlflow.log_artifacts(sampling_cache.cache_dir, "")


if __name__ == "__main__":
    # pylint: disable-next=no-value-for-parameter
    sample()
