import os
from typing import Sequence, Tuple, final

from feature_models.parsing import SplcFmParser
import logging


def _constr_to_clauses(constraints, features):
    final_constraints = []
    for constraint in constraints:
        constraint = constraint.replace("!", "-")
        for id, feature in features.items():
            constraint = constraint.replace(feature, str(id))
        constraint = constraint.replace(" | ", " ")
        constraint += " 0"
        if constraint.count(" ") == 1:
            mand = constraint.split(" ")[0]
            final_constraints += [
                f"{mand} -{str(id)} 0" for id in features.keys() if str(id) != mand
            ]
        final_constraints.append(constraint)

    return list(set(final_constraints))


def _generate_dimacs(binary: Sequence, constraints: Sequence[str]) -> str:
    features = {i + 1: binary[i] for i in range(len(binary))}

    lines = [f"c {str(k)} {v}" for k, v in features.items()]
    clauses = _constr_to_clauses(constraints, features)
    lines += [f"p cnf {len(lines)} {len(clauses)}"]
    lines += clauses
    return "\n".join(lines)


class FeatureModel:
    def __init__(self, xml_file: str):
        self._parser = SplcFmParser()
        self.binary, self.numeric, self.constraints = self._parser.parse(xml_file)
        self.dimacs = _generate_dimacs(self.binary, self.constraints)
        self.xml = self._parser.get_xml()

    def get_features(self):
        return {"binary": self.binary, "numeric": self.numeric}
