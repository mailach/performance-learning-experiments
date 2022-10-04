import tempfile
import os
import json
import logging
import xml.etree.ElementTree as ET

from utils.exceptions import handle_exception


class CacheHandler:

    @handle_exception("Unable to instanciate CacheHandler.")
    def __init__(self, run_id: str, artifact_path: str = None) -> None:
        self._temp_dir = tempfile.gettempdir()
        self.cache_dir = os.path.join(self._temp_dir, run_id)
        self._artifact_path = artifact_path

        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
            self.existing_cache = False
            logging.info(
                f"Successfully created cache directory for run {run_id}")
        else:
            self.existing_cache = True

    @handle_exception("Unable to save artifact to cache.")
    def _save_artifact(self, file: str, artifact: any) -> None:
        logging.info(f"Save artifact {file} to cache...")

        artifact_file = os.path.join(self.cache_dir, self._artifact_path,
                                     file) if self._artifact_path else os.path.join(self.cache_dir, file)

        if ".xml" in artifact_file:
            artifact.write(artifact_file)
        elif ".json" in artifact_file:
            with open(artifact_file, "w") as f:
                json.dump(artifact, f)
        else:
            with open(artifact_file, "w") as f:
                f.write(artifact)

    def save(self, artifacts: dict[str, any]) -> None:
        for id, artifact in artifacts.items():
            self._save_artifact(id, artifact)

    @handle_exception("Unable to retrieve artifact from cache.")
    def _load_artifact(self, file: str) -> any:
        logging.info(f"Retrieve artifact {file} from cache...")
        artifact_file = os.path.join(self.cache_dir, self._artifact_path,
                                     file) if self._artifact_path else os.path.join(self.cache_dir, file)

        if ".xml" in file:
            artifact = ET.parse(artifact_file)
        else:
            with open(artifact_file, "r") as f:
                artifact = json.load(f)
        return artifact

    def retrieve(self, ids: (list | str)) -> any:
        if isinstance(ids, list):
            artifact = {id: self._load_artifact(id) for id in ids}
        else:
            artifact = self._load_artifact(ids)
        return artifact
