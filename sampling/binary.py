import logging
from random import randrange
import z3

from abc import ABC, abstractmethod
from typing import Sequence

from .feature_model import FeatureModel, ConfigurationSolver


def _config_to_str(config: dict[str:int]) -> str:
    return "".join(str(o) for o in config.values())


def _config_is_in_configs(config, configs):
    config = _config_to_str(config)
    configs = [_config_to_str(c) for c in configs]
    return config in configs


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
    def _enable_option(self, config: dict[str, int]):
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
                config = self._enable_option(config)
                if not config:
                    break
                valid = self.cns.is_valid(config)
            if valid:
                configs.append(config)
        return configs


class NegativeOptionWiseSampler(BinarySampler):
    def _disable_options(self, config: dict[str, int]):
        for option in config:
            if config[option] == 1 and not self.cns.mandatory(option):
                config[option] = 0
                return config

    def _search_maximal(literal) -> dict[str, int]:
        config = {str(l): 1 for l in self.cns.literals}
        config[literal] = 0
        enabled = len(self.cns.literals) - 1
        valid = self.cns.valid(config)

    def sample(self, n=None):
        optional = self.cns.optional
        solutions = []
        n_options = len(self.cns.literals)
        size = self.cns.size

        for option in optional:
            opt = z3.Optimize()
            opt.add(self.cns.bitvec)
            for solution in solutions:
                opt.add(solution != size)
            opt.add(z3.Extract(int(option), int(option), size) == 1)

            func = z3.Sum(
                [
                    z3.ZeroExt(n_options, z3.Extract(i, i, size))
                    for i in range(n_options)
                ]
            )

            opt.minimize(func)
            if opt.check() == z3.sat:
                solution = opt.model()[size]
                solutions.append(solution)
        return solutions


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
