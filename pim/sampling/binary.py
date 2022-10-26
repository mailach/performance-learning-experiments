from abc import ABC, abstractmethod
from typing import Sequence
import pandas as pd
from sklearn.model_selection import train_test_split


def _config_to_str(config: dict[str:int]) -> str:
    return "".join(str(o) for o in config.values())


def _config_is_in_configs(config, configs):
    config = _config_to_str(config)
    configs = [_config_to_str(c) for c in configs]
    return config in configs


class Sampler(ABC):
    @abstractmethod
    def sample(self, n: int, all_configs: pd.DataFrame) -> Sequence:
        pass


class TrueRandomSampler(Sampler):
    def sample(self, n: int, all_configs: pd.DataFrame):

        train, test = train_test_split(all_configs, train_size=n)

        return train, test


def SamplerFactory(method: str) -> Sampler:

    sampler = {
        "true_random": TrueRandomSampler,
    }

    return sampler[method]()
