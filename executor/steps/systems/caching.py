import tempfile
import os
import json
import logging
import xml.etree.ElementTree as ET
import pandas as pd
import time
from mlflow.artifacts import download_artifacts


def _handle_xml(filename, artifact=None):
    if artifact:
        artifact.write(filename)
        return None
    return ET.parse(filename)


def _handle_json(filename, artifact=None):
    if artifact:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(artifact, file)
        return None
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def _handle_tsv(filename, artifact=None):
    if artifact is None:
        return pd.read_csv(filename, sep="\t")
    artifact.to_csv(filename, sep="\t", index=False)
    return None


def _handle_dimacs(filename, artifact=None):
    if artifact:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(artifact)
            return None
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


def _file_handling(filename, artifact=None):
    ending = filename.split(".")[-1]
    handlers = {
        "tsv": _handle_tsv,
        "xml": _handle_xml,
        "json": _handle_json,
        "dimacs": _handle_dimacs,
    }
    return handlers[ending](filename, artifact)


class CacheHandler:
    """
    Handles interactions with temporary directory of the local filesystems from within runs.
    Further handles caching for runs stored remotely.

    ...

    Attributes
    ----------
    cache_dir : str
        the temporary directory maintained by this instance of CacheHandler class

    Methods
    -------
    __init__(run_id, new_run):
        Constructor, generates a CacheHandler instance

    save(artifacts):
        Takes artifacts and saves them in resp. directory

    retrieve(ids):
        Takes and id or a list of ids and returns the resp. artifacts from the cache.
    """

    def __init__(self, run_id: str, new_run: bool = True) -> None:
        """
        Constructor: instantiates CacheHandler for run.

        Parameters
        ----------
        run_id : str
            the run_id connected to this CacheHandler instance
        new_run: bool
            whether the cachehandler for this run is a new run.

        """
        self._temp_dir = tempfile.gettempdir()
        self.cache_dir = os.path.join(self._temp_dir, run_id)
        if new_run:
            self._generate_cache()
        elif not os.path.exists(self.cache_dir):
            self._generate_cache()
            self._load_artifacts_from_remote(run_id)
        else:
            logging.info("Use existing cache for run %s", run_id)

    def _generate_cache(self):
        logging.info("Create cache directory for run %s", self.cache_dir)
        os.mkdir(self.cache_dir)
        self.existing_cache = False
        logging.info("Successfully created cache directory for run %s", self.cache_dir)

    def _load_artifacts_from_remote(self, run_id: str):
        logging.info("Downloading artifacts of run %s to cache...", run_id)
        download_artifacts(run_id=run_id, dst_path=self.cache_dir)
        logging.info("Download successful...")

    def save(self, artifacts: dict) -> None:
        """
        Takes artifacts and saves them.

        Parameters
        ----------
        artifacts : dict[str, any]
            Dictionary where keys are filenames and values are artifacts for storage.
        """
        for name, artifact in artifacts.items():
            filename = os.path.join(self.cache_dir, name)
            _file_handling(filename, artifact)

    def _load_artifact(self, filename: str) -> any:

        filename = os.path.join(self.cache_dir, filename)
        try:
            logging.info("Retrieve artifact %s from cache...", filename)
            artifact = _file_handling(filename)
        except:
            logging.error("Can not retrieve artifact %s", filename)
            time.sleep(120)
            raise Exception
        logging.info(" Succssesfully retrieved artifact %s from cache...")
        return artifact

    def retrieve(self, filenames) -> any:
        """
        Takes filenames and returns the corresponding artifacts.

        Parameters
        ----------
        filenames : list | str
            filenames
        """
        if isinstance(filenames, list):
            artifacts = []
            for filename in filenames:
                filename = os.path.join(self.cache_dir, filename)
                artifacts.append(_file_handling(filename))
                return artifacts
        artifact = self._load_artifact(filenames)
        return artifact
