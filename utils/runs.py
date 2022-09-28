import logging
import tempfile

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils import mlflow_tags
from mlflow.entities import Run, Experiment
from mlflow.artifacts import download_artifacts


def _find_or_create_exp_id(entrypoint, client):
    exp_id = mlflow.search_experiments(filter_string=f"name = '{entrypoint}'")
    return exp_id[0] if exp_id else client.create_experiment(entrypoint)


def _same_run(run: Run, params: dict[str:any]):

    same_parameter = [
        True if run.data.params.get(param) == str(params[param]) else False
        for param in params.keys()
        if param != "cache_dir"
    ]

    return True if all(same_parameter) else False


def _generate_filter_string(params: dict[str, any]):
    clauses = [
        "parameter." + param + " = '" + str(value) + "'"
        for param, value in params.items()
        if param != "data_dir"
    ]
    query = " AND ".join(clauses) + " AND attribute.status = 'FINISHED'"
    return query


def get_run_if_exists(
    entrypoint: str, parameters: dict[str, any], git_commit: str, client: MlflowClient
):
    filter_string = _generate_filter_string(parameters)
    runs = mlflow.search_runs(
        experiment_names=[entrypoint], filter_string=filter_string
    )
    return runs["run_id"][0] if not runs.empty else False
