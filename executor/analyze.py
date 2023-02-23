import pandas as pd
import mlflow
from itertools import product


from executor.parsing import Executor


def _generate_iv_combinations(ivs):
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


def analyze(executor: Executor):
    iv_steps = {
        k: v for k, v in executor.config["parametrization"].items() if k != "type"
    }
    ivs = _extract_ivs(iv_steps)
    return [c for c in _generate_iv_combinations(ivs)]
