from operator import index
import xml.etree.ElementTree as ET
from typing import Sequence
import pandas as pd




def features_from_dimacs(dimacs_file: str) -> dict[int, str]:

    with open(dimacs_file, "r") as f:
        features = {int(line.split()[1]): line.split()[2].strip() for line in f.readlines() if line[0]=="c"}

    return features


def _check_feature_existence(config: Sequence[str], features: Sequence[str]) -> None:
    not_matching = [feature for feature in config if feature not in features]
    if len(not_matching):
        raise Exception(f"Can not convert file because '{not_matching}' is/are not listed in the feature model")


def _one_hot_encode(config: Sequence[str], features: Sequence[str], value: str):
    _check_feature_existence(config, features)
    oh = {feature: 1 if feature in config else 0 for feature in features}
    oh["measured_value"] = value

    return oh

def xml_measurements_to_onehot(input_file: str, dimacs_file: str, output_file: str, shema: str) -> None:
    features = features_from_dimacs(dimacs_file)
    one_hot = []

    df = ET.parse(input_file).getroot()
    for row in df:
        try: 
            config = [f.strip() for f in row.find(f'.//data[@columname="Configuration"]').text.split(",") if f.strip() != ""]
            one_hot.append(_one_hot_encode(config, features.values(), row.find(f'.//data[@columname="Measured Value"]').text))
        except Exception as e:
            print(row[0].attrib, row[0].text)
            print(row[1].attrib, row[1].text)
            raise e


    pd.DataFrame(one_hot).to_csv(output_file, sep="\t", index=False)


            






