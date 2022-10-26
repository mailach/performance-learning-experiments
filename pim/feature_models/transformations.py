from typing import Sequence
import pandas as pd
from pim.feature_models.parsing import SplcMeasurementParser


def _check_feature_existence(config: Sequence[str], features: Sequence[str]) -> None:
    not_matching = [feature for feature in config if feature not in features]
    if len(not_matching):
        raise Exception(
            f"Can not convert file because '{not_matching}' is/are not listed in the feature model"
        )


def _extract_numeric(measurement, numerics):
    transformed = {}
    measurement = {
        m.split(";")[0]: m.split(";")[1] for m in measurement["numerics"].split(",")
    }

    _check_feature_existence(measurement.keys(), numerics)
    for numeric in numerics:
        transformed[numeric] = (
            float(measurement[numeric]) if numeric in measurement else None
        )
    return transformed


def _extract_binary(measurement, binaries):
    transformed = {}
    _check_feature_existence(measurement["binaries"].strip(",").split(","), binaries)
    for binary in binaries:
        transformed[binary] = 1 if binary in measurement["binaries"] else 0
    return transformed


def _extract_nfp(measurement):
    transformed = {}
    for name, value in measurement["nfp"].items():
        transformed[name] = float(value)

    return transformed


def _measurements_to_df(measurements, binaries, numerics):
    table = []
    for measurement in measurements:
        transformed = {}
        transformed.update(_extract_binary(measurement, binaries))
        transformed.update(_extract_numeric(measurement, numerics))
        transformed.update(_extract_nfp(measurement))
        table.append(transformed)
    return pd.DataFrame(table)


class Measurements:
    """
    Measurements corresponding to a specific featuremodel.
    ...

    Attributes
    ----------
    measurements : Sequence
        list of rows, represented as dictionaries
    df : pd.DataFrame
        pandas representation of measurements
    xml : xml.etree.ElementTree
        xml representation of measurements
    """

    def __init__(self, filename: str, binary, numeric):
        self._parser = SplcMeasurementParser()
        self.measurements = self._parser.parse(filename)
        self.df = _measurements_to_df(self.measurements, binary, numeric)
        self.xml = self._parser.get_xml()
