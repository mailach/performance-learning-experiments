import xml.etree.ElementTree as ET
from typing import Sequence, Tuple
from abc import ABC, abstractmethod


class Fm_parser(ABC):
    @abstractmethod
    def parse():
        pass


class Splc_parser(Fm_parser):
    def _ext_opt_constr(self, element_list: ET.Element):
        return [el.text for el in element_list]

    def _ext_opt(self, element_list: ET.Element):
        opts = element_list.findall("configurationOption")
        result = []
        for opt in opts:
            o = {
                p.tag: p.text.strip() if isinstance(p.text, str) else p.text
                for p in opt
            }
            o["impliedOptions"] = (
                self._ext_opt_constr(opt.find("impliedOptions"))
                if "impliedOptions" in o
                else []
            )

            o["excludedOptions"] = (
                self._ext_opt_constr(opt.find("excludedOptions"))
                if "excludedOptions" in o
                else []
            )
            result.append(o)

        return result

    def _ext_constr(self, element_list: ET.Element):
        constrs = element_list.findall("constraint")
        return [p.text.strip() if isinstance(p.text, str) else p.text for p in constrs]

    def _ext_data(self, xml_tree: ET):
        vm = xml_tree.getroot()

        otags = ["binaryOptions", "numericOptions"]
        for opt in otags:
            yield self._ext_opt(vm.find(opt))

        ctags = ["booleanConstraints", "nonBooleanConstraints", "mixedConstraints"]
        for constr in ctags:
            yield self._ext_constr(vm.find(constr))

    def _constraints_from_options(self, options):
        constraints = []
        for o in options:
            if len(o["impliedOptions"]):
                constraints += [f"!{o['name']} | {io}" for io in o["impliedOptions"]]
            if len(o["excludedOptions"]):
                constraints += [f"!{o['name']} | {io}" for io in o["excludedOptions"]]

            if o["optional"] == "False":
                constraints += [o["name"]]

        return constraints

    def _transform_options(self, binOpt, numOpt):
        constraints = self._constraints_from_options(binOpt)
        features = {}
        for i in range(1, len(binOpt) + 1):
            features[i] = {"name": binOpt[i - 1]["name"], "type": "bin"}
        for i in range(len(binOpt) + 1, len(binOpt) + 1):
            o = numOpt[i - len(binOpt) + 1]
            features[i] = {
                "name": o["name"],
                "type": "num",
                "min": o["minValue"],
                "max": o["maxValue"],
                "step": o["stepFunction"],
            }
        return features, constraints

    def parse(self, xml_tree: ET):
        binOpt, numOpt, boolCon, nboolCon, mixedCon = self._ext_data(xml_tree)
        features, constraints = self._transform_options(binOpt, numOpt)
        constraints += boolCon + nboolCon + mixedCon
        return features, constraints


class Sxfm_parser(Fm_parser):
    def __init__():
        raise NotImplementedError

    def parse(self, xml_tree: ET):
        pass


def ParserFactory(format: str):
    parsers = {"splc": Splc_parser, "sxml": Sxfm_parser}

    return parsers[format]()
