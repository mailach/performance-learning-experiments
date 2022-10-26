from typing import Sequence


class FeatureModel:
    def __init__(self, features: dict[str, str], constraints=Sequence):
        self.features = features
        self.constraints = constraints

    def to_dimacs(self) -> str:
        dimacs = "".join([f"c {id} {name}" for id, name in self.features.items()])
        dimacs += "\n" + f"p cnf {len(self.features)} {len(self.constraints)}\n"
        dimacs += "\n".join(self._constraints_to_clauses())
        return dimacs

    def _constraints_to_clauses():
        pass
