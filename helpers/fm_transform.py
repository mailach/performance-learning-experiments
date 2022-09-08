## Feature model parser from xml
from itertools import combinations
import xml.etree.ElementTree as ET
from typing import Sequence, Tuple




def __alternative(literals: Sequence[int]) -> Sequence[Sequence[int]]:
    return [literals] + [[-a, -b] for a,b in combinations(literals, 2)]

def __lor(literals: Sequence[int]) -> Sequence[Sequence[int]]:
    return [literals]

def __a_requires_b(a: int, b: int) -> Sequence[Sequence[int]]:
    return [[-a, b]]

def __a_excludes_b(a: int, b: int) -> Sequence[Sequence[int]]:
    return [[-a, -b]]

def __commulative(literals: Sequence[int]) -> Sequence[Sequence[int]]:
    raise Exception("Commulative constraint not implemented yet")



def __extract_features_schema2015(root)-> Tuple[dict[int, str], Sequence[Sequence[int]]]:
    clauses = []

    features = {}

    for element in root:
        id = int(element.attrib["id"])
        features[id] = element.attrib["name"]
        if element.attrib["optional"] == "false" and not len(element.find("parentElement")):
            clauses.append([id])

        constraints = element.find("constraints")  
        for constraint in constraints:

            if len(constraint):

                if constraint.attrib["type"] == "alternative":
                    
                    parent_optional = root.find(f'.//element[@id="{element.find("parentElement")[0].text}"]').attrib["optional"]
                    
                    if parent_optional == "false":
                        for clause in __alternative([id] + [int(c.find("id").text) for c in constraint]):
                            if sorted(clause) not in clauses:
                                clauses.append(sorted(clause))
                    else:
                        for clause in __alternative([id] + [int(c.find("id").text) for c in constraint]):
                            if sorted(clause) not in clauses:
                                clauses.append(sorted(clause))    
                
                elif constraint.attrib["type"] == "requires":
                    for required in constraint:
                        clauses += __a_requires_b(id, int(required.find("id").text))


                elif constraint.attrib["type"] == "excludes":
                    for excluded in constraint:
                        clauses += __a_excludes_b(id, int(excluded.find("id").text))
    
    return features, clauses

def __extract_features(root: ET, shema_identifier: str) -> Tuple[dict[int, str], Sequence[Sequence[int]]]:


    if shema_identifier=="shema2015":
        features, clauses = __extract_features_schema2015(root)
    else:
        raise Exception(f"Conversion not implemented for sheme {shema_identifier}")
    return features, clauses



def __generate_dimacs(features: dict[int, str], clauses: Sequence[Sequence[int]]) -> str:
    lines = [f"c {str(k)} {v}" for k,v in features.items()]
    lines += [f"p cnf {len(features)} {len(clauses)}"]
    lines += [" ".join([str(c) for c in clause]) + " 0" for clause in clauses]
    return "\n".join(lines)



def fm_xml_to_dimacs(input_file: str, output_file: str, shema_identifier: str) -> None:
    
    tree = ET.parse(input_file)
    root = tree.getroot()

    features, clauses = __extract_features(root, shema_identifier)
    dimacs = __generate_dimacs(features, clauses)

    with open(output_file, "w") as f:
        f.write(dimacs)

