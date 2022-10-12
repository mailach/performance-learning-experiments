import logging
from random import randrange

from abc import ABC, abstractmethod
from typing import Sequence


class Sampler(ABC):

    @abstractmethod
    def sample(self, n: int, all_configs: list = None) -> Sequence:
        pass


class TrueRandomSampler(Sampler):

    def sample(self, n: int, all_configs: list):
        self.configs = all_configs
        confs = self.configs.copy()
        if len(confs) < n:
            logging.error(
                f"Desired sample size n={n} is smaller than number of available configurations n={len(confs)}. "
            )
            raise Exception("Valueerror for samplesize.")

        sampled = [confs.pop(randrange(len(confs)))
                   for _ in range(n)]

        return sampled, confs


class BinaryOptionSampler(Sampler):

    @abstractmethod
    def load_fm():
        pass


def SamplerFactory(method: str) -> Sampler:

    sampler = {
        "true_random": TrueRandomSampler,
    }

    return sampler[method]()
