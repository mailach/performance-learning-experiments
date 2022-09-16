import json
import os
import xml.etree.ElementTree as ET
from typing import Sequence
import pandas as pd


def features_from_dimacs(path: str) -> dict[int, str]:

    with open(os.path.join(path, "fm_cnf.dimacs"), "r") as f:
        features = {
            int(line.split()[1]): line.split()[2].strip()
            for line in f.readlines()
            if line[0] == "c"
        }

    return features


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


def xml_measurements_to_onehot(path: str) -> None:
    features = features_from_dimacs(path)
    one_hot = []

    df = ET.parse(os.path.join(path, "all_measurements.xml")).getroot()
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

    with open(os.path.join(path, "measurements.json"), "w") as f:
        json.dump(one_hot, f)
    with open(os.path.join(path, "features.json"), "w") as f:
        json.dump(features, f)

    # pd.DataFrame(one_hot).to_csv(output_file, sep="\t", index=False)
