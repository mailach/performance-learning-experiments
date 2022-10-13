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
        return self.cns.generate_configurations(n)


class OptionWiseSampler(BinarySampler):
    def _add_enabled_option(self, config: dict[str, int]):
        for option in config:
            if config[option] == 0:
                config[option] = 1
                return config

    def sample(self, n=None):
        configs = []

        for literal in self.cns.literals:
            config = self.cns.get_minimal()
            config[str(literal)] = 1
            valid = self.cns.is_valid(config)
            while not valid:
                config = self._add_enabled_option(config)
                if not config:
                    break
                valid = self.cns.is_valid(config)
            if valid:
                configs.append(config)
        return configs


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
