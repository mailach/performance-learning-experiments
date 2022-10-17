## Feature model parser from xml
from calendar import c
import os
from xmlschema import validate
from itertools import combinations
import xml.etree.ElementTree as ET
from typing import Sequence, Tuple, final

from feature_models.parsing import ParserFactory


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


def _constr_to_clauses(constraints, features):
    final_constraints = []
    print(features)
    for constraint in constraints:
        constraint = constraint.replace("!", "-")
        for id, feat in features.items():
            constraint = constraint.replace(feat["name"], str(id))
        constraint = constraint.replace(" | ", " ")
        constraint += " 0"
        if constraint.count(" ") == 1:
            mand = constraint.split(" ")[0]
            final_constraints += [
                f"{mand} -{str(id)} 0" for id in features.keys() if str(id) != mand
            ]
        final_constraints.append(constraint)

    return list(set(final_constraints))


def _generate_dimacs(features: dict[int, str], constranints: Sequence[str]) -> str:

    lines = [
        f"c {str(k)} {v['name']}" for k, v in features.items() if v["type"] == "bin"
    ]
    clauses = _constr_to_clauses(constranints, features)
    lines += [f"p cnf {len(lines)} {len(clauses)}"]
    lines += clauses
    return "\n".join(lines)


def _get_schema(file):
    try:
        validate(file, "data/schema/splc.xsd")
        return "splc"
    except:
        raise Exception


class Fm_handler:
    def __init__(self, data_dir: str):
        xml_file = os.path.join(data_dir, "fm.xml")
        parser = ParserFactory(_get_schema(xml_file))
        self.xml = ET.parse(xml_file)
        self.features, self.constraints = parser.parse(self.xml)
        self.binary = {
            id: v["name"] for id, v in self.features.items() if v["type"] == "bin"
        }
        self.numeric = {
            id: v["name"] for id, v in self.features.items() if v["type"] == "num"
        }

        self.dimacs = _generate_dimacs(self.features, self.constraints)

    def get_properties(self):
        return {"fm.xml": self.xml, "fm.dimacs": self.dimacs}
