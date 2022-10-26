from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET
import xmlschema
import logging
import os


def _implication(option1, options):
    return [f"!{option1} | {opt}" for opt in options]


def _exclusion(option1, options, optional=None):
    simple_exclusion = [f"!{option1} | !{opt}" for opt in options]
    if optional == "True":
        return simple_exclusion
    else:
        return [" | ".join([option1] + options)] + simple_exclusion


class Parser(ABC):
    schema: xmlschema.XMLSchema = None
    decoded_xml: dict = None

    def _validate_and_decode(self, xml_file: str):
        try:
            self.schema.validate(xml_file)
        except Exception as e:
            logging.error("The provided xml file is not valid vm format.")
            raise e
        self.decoded_xml = self.schema.decode(xml_file)

    def get_xml(self):
        return ET.ElementTree(self.schema.encode(self.decoded_xml))

    @abstractmethod
    def parse():
        pass


class FmParser(Parser):
    @abstractmethod
    def _extract_binaries(self):
        pass

    @abstractmethod
    def _extract_bool_constraints(self):
        pass


class SplcFmParser(FmParser):
    def __init__(self):
        self.schema = xmlschema.XMLSchema("data/schema/splc_fm.xsd")

    def _extract_binaries(self):
        binaries = []
        constraints = []
        binaryOptions = self.decoded_xml["binaryOptions"]["configurationOption"]
        for bo in binaryOptions:
            binaries.append(bo["name"])
            if bo["impliedOptions"]:
                constraints += _implication(bo["name"], bo["impliedOptions"]["option"])
            if bo["excludedOptions"]:
                constraints += _exclusion(
                    bo["name"], bo["excludedOptions"]["option"], bo["optional"]
                )
        return binaries, constraints

    def _extract_numerics(self):
        numerics = []
        try:
            numericOptions = self.decoded_xml["numericOptions"]["configurationOption"]
        except:
            return numerics

        for no in numericOptions:
            numerics.append(no["name"])
        return numerics

    def _extract_bool_constraints(self):
        if self.decoded_xml["booleanConstraints"]:
            return self.decoded_xml["booleanConstraints"]
        else:
            return []

    def parse(self, xml_file: str):
        self._validate_and_decode(xml_file)
        binaries, constraints = self._extract_binaries()
        numerics = self._extract_numerics()
        constraints += self._extract_bool_constraints()
        return binaries, numerics, constraints


class MeasurementParser(Parser):
    @abstractmethod
    def _extract_rows(self):
        pass


class SplcMeasurementParser(MeasurementParser):
    def __init__(self):
        self.schema = xmlschema.XMLSchema("data/schema/splc_measurements.xsd")

    def _extract_rows(self):
        rows = []
        for row in self.decoded_xml["row"]:
            config = {"nfp": {}}
            for column in row["data"]:
                if column["@column"] == "Configuration":
                    config["binaries"] = column["$"].replace("\n", "")
                elif column["@column"] == "Variable Features":
                    config["numerics"] = column["$"].replace("\n", "")
                else:
                    config["nfp"][column["@column"]] = column["$"].replace("\n", "")
            rows.append(config)

        return rows

    def parse(self, xml_file):
        self._validate_and_decode(xml_file)
        return self._extract_rows()
