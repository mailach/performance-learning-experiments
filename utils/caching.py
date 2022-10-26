import tempfile
import os
import json
import logging
import xml.etree.ElementTree as ET
import pandas as pd

from mlflow.artifacts import download_artifacts

from utils.exceptions import handle_exception


def _handle_xml(filename, artifact=None):
    if artifact:
        artifact.write(filename)
    else:
        return ET.parse(filename)


def _handle_json(filename, artifact=None):
    if artifact:
        with open(filename, "w") as f:
            json.dump(artifact, f)
    else:
        with open(filename, "r") as f:
            return json.load(f)


def _handle_tsv(filename, artifact=None):
    if artifact is None:
        return pd.read_csv(filename, sep="\t")
    else:
        artifact.to_csv(filename, sep="\t", index=False)


def _handle_dimacs(filename, artifact=None):
    if artifact:
        with open(filename, "w") as f:
            f.write(artifact)
    else:
        with open(filename, "r") as f:
            return f.read()


def _fileHandling(filename, artifact=None):
    ending = filename.split(".")[-1]
    handlers = {
        "tsv": _handle_tsv,
        "xml": _handle_xml,
        "json": _handle_json,
        "dimacs": _handle_dimacs,
    }
    return handlers[ending](filename, artifact)


class CacheHandler:
    def __init__(self, run_id: str, new_run: bool = True) -> None:
        self._temp_dir = tempfile.gettempdir()
        self.cache_dir = os.path.join(self._temp_dir, run_id)
        if new_run:
            self._generate_cache()
        elif not os.path.exists(self.cache_dir):
            self._generate_cache()
            self._load_artifacts_from_remote(run_id)
        else:
            logging.info(f"Use existing cache for run {run_id}")

    def _generate_cache(self):
        logging.info(f"Create cache directory for run {self.cache_dir}")
        os.mkdir(self.cache_dir)
        self.existing_cache = False
        logging.info(f"Successfully created cache directory for run {self.cache_dir}")

    def _load_artifacts_from_remote(self, run_id: str):
        logging.info(f"Downloading artifacts of run {run_id} to cache...")
        download_artifacts(run_id=run_id, dst_path=self.cache_dir)
        logging.info(f"Download successful...")

    def save(self, artifacts: dict[str, any]) -> None:
        for id, artifact in artifacts.items():
            filename = os.path.join(self.cache_dir, id)
            _fileHandling(filename, artifact)

    def _load_artifact(self, filename: str) -> any:
        logging.info(f"Retrieve artifact {filename} from cache...")
        filename = os.path.join(self.cache_dir, filename)
        return _fileHandling(filename)

    def retrieve(self, ids: (list | str)) -> any:
        if isinstance(ids, list):
            artifacts = {}
            for id in ids:
                filename = os.path.join(self.cache_dir, id)
                artifacts[id] = _fileHandling(filename)
            return artifacts
        else:
            artifact = self._load_artifact(ids)
            return artifact
