import logging
from random import randrange

from abc import ABC, abstractmethod
from typing import Sequence

from .feature_model import FeatureModel, ConfigurationSolver


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


class BinarySampler(Sampler):

    def __init__(self, fm: FeatureModel):
        self.fm = fm
        self.cns = ConfigurationSolver(fm.constraints, fm.features.keys())

    @abstractmethod
    def sample(n: int):
        pass


class PseudoRandomSampler(BinarySampler):

    def sample(self, n: int):
        return cns.generate_configurations(n)


class OptionWiseSampler(BinarySampler):
    def sample(n: int):
        pass


class NegativeOptionWiseSampler(BinarySampler):
    def __init__(self, fm: FeatureModel):
        raise NotImplementedError

    def sample(n: int):
        pass


def BinarySamplerFactory(method: str, fm: FeatureModel):
    samplers = {
        "pr": PseudoRandomSampler,
        "ow": OptionWiseSampler,
        "now": NegativeOptionWiseSampler, }
    return samplers[method](fm)


def SamplerFactory(method: str) -> Sampler:

    sampler = {
        "true_random": TrueRandomSampler,
    }

    return sampler[method]()
