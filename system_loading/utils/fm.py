## Feature model parser from xml
import os
from itertools import combinations
import xml.etree.ElementTree as ET
from typing import Sequence, Tuple


def _alternative(literals: Sequence[int]) -> Sequence[Sequence[int]]:
    return [literals] + [[-a, -b] for a, b in combinations(literals, 2)]


def _lor(literals: Sequence[int]) -> Sequence[Sequence[int]]:
    return [literals]


def _a_requires_b(a: int, b: int) -> Sequence[Sequence[int]]:
    return [[-a, b]]


def _a_excludes_b(a: int, b: int) -> Sequence[Sequence[int]]:
    return [[-a, -b]]


def _commulative(literals: Sequence[int]) -> Sequence[Sequence[int]]:
    raise Exception("Commulative constraint not implemented yet")


def _extract_features_schema2015(
    root,
) -> Tuple[dict[int, str], Sequence[Sequence[int]]]:
    clauses = []

    features = {}
    for element in root.findall("element"):
        id = int(element.attrib["id"])
        features[id] = element.attrib["name"]
        if element.attrib["optional"] == "false" and not len(
            element.find("parentElement")
        ):
            clauses.append([id])

        constraints = element.find("constraints")
        for constraint in constraints:

            if len(constraint):

                if constraint.attrib["type"] == "alternative":

                    parent_optional = root.find(
                        f'.//element[@id="{element.find("parentElement")[0].text}"]'
                    ).attrib["optional"]

                    if parent_optional == "false":
                        for clause in _alternative(
                            [id] + [int(c.find("id").text) for c in constraint]
                        ):
                            if sorted(clause) not in clauses:
                                clauses.append(sorted(clause))
                    else:
                        for clause in _alternative(
                            [id] + [int(c.find("id").text) for c in constraint]
                        ):
                            if sorted(clause) not in clauses:
                                clauses.append(sorted(clause))

                elif constraint.attrib["type"] == "requires":
                    for required in constraint:
                        clauses += _a_requires_b(id, int(required.find("id").text))

                elif constraint.attrib["type"] == "excludes":
                    for excluded in constraint:
                        clauses += _a_excludes_b(id, int(excluded.find("id").text))

    return features, clauses


def _extract_features(
    root: ET, shema_identifier: str
) -> Tuple[dict[int, str], Sequence[Sequence[int]]]:

    if shema_identifier == "shema2015":
        features, clauses = _extract_features_schema2015(root)
    else:
        raise Exception(f"Conversion not implemented for sheme {shema_identifier}")
    return features, clauses


def _generate_dimacs(features: dict[int, str], clauses: Sequence[Sequence[int]]) -> str:
    lines = [f"c {str(k)} {v}" for k, v in features.items()]
    lines += [f"p cnf {len(features)} {len(clauses)}"]
    lines += [" ".join([str(c) for c in clause]) + " 0" for clause in clauses]
    return "\n".join(lines)


def fm_xml_to_dimacs(path: str, shema_identifier: str) -> None:

    tree = ET.parse(os.path.join(path, "fm.xml"))
    root = tree.getroot()

    features, clauses = _extract_features(root, shema_identifier)
    dimacs = _generate_dimacs(features, clauses)

    with open(os.path.join(path, "fm_cnf.dimacs"), "w") as f:
        f.write(dimacs)
