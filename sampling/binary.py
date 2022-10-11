import logging
import random

from abc import ABC, abstractmethod
from typing import Sequence


class Sampler(ABC):
    @abstractmethod
    def load_fm(self, fm: str = None, all_configs: list = None):
        pass

    @abstractmethod
    def sample(self, n: int) -> Sequence:
        pass


class TrueRandomSampler(Sampler):
    def load_fm(self, all_configs: list):
        self.configs = all_configs

    def sample(self, n: int):
        val_configs = self.configs.copy()
        if len(val_configs) < n:
            logging.error(
                f"Desired sample size n={n} is smaller than number of available configurations n={len(val_configs)}. "
            )
            raise Exception("Valueerror for samplesize.")

        sampled = [
            self.configs.pop(random.randrange(len(val_configs))) for _ in range(n)
        ]

        return sampled, val_configs


def SamplerFactory(method: str) -> Sampler:

    sampler = {
        "true_random": TrueRandomSampler,
    }

    return sampler[method]()
