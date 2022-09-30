import os
import xml.etree.ElementTree as ET
from typing import Sequence


def _check_feature_existence(config: Sequence[str], features: Sequence[str]) -> None:
    not_matching = [feature for feature in config if feature not in features]
    if len(not_matching):
        raise Exception(
            f"Can not convert file because '{not_matching}' is/are not listed in the feature model"
        )


def _one_hot_encode(config: Sequence[str], features: Sequence[str], value: str):
    _check_feature_existence(config, features.values())
    oh = {key: 1 if feature in config else 0 for key, feature in features.items()}
    oh["measured_value"] = float(value.replace(",", "."))

    return oh


def _xml_measurements_to_onehot(
    data_tree: ET, features: dict[int, str]
) -> Sequence[dict[str, int]]:

    one_hot = []

    df = data_tree.getroot()
    for row in df:
        try:
            config = [
                f.strip()
                for f in row.find(f'.//data[@columname="Configuration"]').text.split(
                    ","
                )
                if f.strip() != ""
            ]
            one_hot.append(
                _one_hot_encode(
                    config,
                    features,
                    row.find(f'.//data[@columname="Measured Value"]').text,
                )
            )
        except Exception as e:
            print(row[0].attrib, row[0].text)
            print(row[1].attrib, row[1].text)
            raise e

    return one_hot


class Measurement_handler:
    def __init__(self, data_dir: str, features: dict[int, str]):
        self.xml = ET.parse(os.path.join(data_dir, "all_measurements.xml"))
        self.one_hot = _xml_measurements_to_onehot(self.xml, features)
