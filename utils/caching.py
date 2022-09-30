import tempfile
import os
import json
import logging
import xml.etree.ElementTree as ET

from utils.exceptions import handle_exception


class CacheHandler:
    items = {}

    @handle_exception("Unable to instanciate CachHandler.")
    def __init__(self, run_id: str) -> None:
        self._temp_dir = tempfile.gettempdir()
        self.cache_dir = os.path.join(self._temp_dir, run_id)

        os.mkdir(self.cache_dir)
        logging.info(f"Successfully created cache directory for run {run_id}")

    @handle_exception("Unable to save artifact to cache.")
    def _save_artifact(self, file: str, artifact: any) -> None:
        logging.info("Save artifact to cache...")
        artifact_file = os.path.join(self.cache_dir, file)

        if ".xml" in artifact_file:
            artifact.write(artifact_file)
        elif ".json" in artifact_file:
            with open(artifact_file, "w") as f:
                json.dump(artifact, f)
        else:
            with open(artifact_file, "w") as f:
                f.write(artifact)

    @handle_exception("Unable to retrieve artifact from cache.")
    def _load_artifact(self, file: str) -> any:
        logging.info("Retrieve artifact from cache...")
        artifact_file = os.path.join(self.cache_dir, file)

        if ".xml" in file:
            return ET.parse(artifact_file)
        else:
            with open(artifact_file, "r") as f:
                return json.load(f)

    def save(self, artifacts: dict[str, any]) -> None:
        for id, artifact in artifacts.items():
            self._save_artifact(id, artifact)

    def retrieve(self, identifiers: list) -> dict[str, any]:
        return {id: self._load_artifact(id) for id in identifiers}
