from typing import Literal, Sequence

import mlflow


def update_metrics(run_id: str, metric: dict[str, any]):
    """
    Updates the metrics of an already existing run.

    Parameters
    ----------
    run_id : str
        run to update
    metric : dict[str, any]
        metrics to update
    """
    with mlflow.start_run(run_id):
        mlflow.log_metrics(metric)


def update_params(run_id: str, params: dict[str, any]):
    """
    Updates the parameters of an already existing run.

    Parameters
    ----------
    run_id : str
        run to update
    params : dict[str, any]
        params to update
    """
    with mlflow.start_run(run_id):
        mlflow.log_params(params)


def _find_or_create_exp_id(entrypoint, client):
    exp_id = mlflow.search_experiments(filter_string=f"name = '{entrypoint}'")
    return exp_id[0] if exp_id else client.create_experiment(entrypoint)


def _same_run(run, params: dict[str:any]):

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
    entrypoint: str, parameters: dict[str, any]
) -> (str | Literal[False]):
    """
    Returns a run if one exists.

    Parameters
    ----------
    parameters : dict[str, any]
        the parameters that the run should contain
    """

    filter_string = _generate_filter_string(parameters)
    runs = mlflow.search_runs(
        experiment_names=[entrypoint], filter_string=filter_string
    )
    return runs["run_id"][0] if not runs.empty else False


def get_all_runs(
    entrypoint: str, parameters: dict[str, any]
) -> (Sequence[str] | Literal[False]):
    """
    Searches for runs with specific parameters and returns either list or false.

    Parameters
    ----------
    parameters : dict[str, any]
        the parameters that the runs should contain
    """

    filter_string = _generate_filter_string(parameters)
    runs = mlflow.search_runs(
        experiment_names=[entrypoint], filter_string=filter_string
    )
    return list(runs["run_id"]) if not runs.empty else False
