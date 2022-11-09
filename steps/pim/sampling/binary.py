from abc import ABC, abstractmethod
from typing import Sequence
import pandas as pd
from sklearn.model_selection import train_test_split
from splc2py.sampling import Sampler


def _config_to_str(config: dict[str:int]) -> str:
    return "".join(str(o) for o in config.values())


def _config_is_in_configs(config, configs):
    config = _config_to_str(config)
    configs = [_config_to_str(c) for c in configs]
    return config in configs


def true_random_sampling(n: int, all_configs: pd.DataFrame):

    train, test = train_test_split(all_configs, train_size=n)

    return train, test


def mixed_sampling(binary_method, numeric_method, vm):
    sampler = Sampler(vm)
    return sampler.sample(binary=binary_method, numeric=numeric_method, format="dict")


def binary_sampling(method, vm):
    sampler = Sampler(vm)
    return sampler.sample(binary=method, format="dict")
