import pandas as pd
import mlflow
from itertools import product


from executor.parsing import Executor


def _generate_iv_combinations(iv_steps):
    ivs = _extract_ivs(iv_steps)
    for combination in product(*ivs.values()):
        yield dict(zip(ivs.keys(), combination))


def _extract_ivs(config):
    iv_levels = {}
    for step, ivs in config.items():
        for iv, levels in ivs.items():
            iv_levels[step + "." + iv] = levels

    return iv_levels


def _load_data_from_mlflow(run_ids):
    rows = []
    for idx in run_ids:
        data = mlflow.get_run(idx).data
        row = {}
        row.update(data.params)
        row.update(data.metrics)
        rows.append(row)
    return rows


def _analyze_data(iv_steps, run_ids, metrics):
    all_data = pd.DataFrame(_load_data_from_mlflow(run_ids))
    iv_combinations = _generate_iv_combinations(iv_steps)

    return _aggregated_data(all_data, iv_combinations, metrics)


def _generate_query_string(combi):
    return " & ".join([f"`{iv}` == '{level}'" for iv, level in combi.items()])


def _calculate_stats(data, ivs, metrics):
    stats = {}
    for iv in ivs:
        stats[iv] = data[iv].iloc[0]

    for metric in metrics:
        tmp_stats = data[metric].describe()
        stats.update({f"{metric}.{k}": v for k, v in tmp_stats.items()})

    return stats


def _aggregated_data(all_data, iv_combinations, metrics):
    aggregated = []
    for combi in iv_combinations:
        ivs = list(combi.keys())
        query_string = _generate_query_string(combi)
        tmp = all_data.query(query_string)
        # return _generate_query_string(combi)
        aggregated.append(_calculate_stats(tmp, ivs, metrics))
    return aggregated


def analyze(executor: Executor, metrics=[]):
    iv_steps = {
        k: v for k, v in executor.config["parametrization"].items() if k != "type"
    }

    run_ids = []
    for _, ids in executor.run_ids.items():
        run_ids += ids

    return _analyze_data(iv_steps, run_ids, metrics)
