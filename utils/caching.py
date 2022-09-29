import tempfile
import os
import json
import logging

from exceptions import handle_exception


@handle_exception("Unable to save artifact to cache.")
def _save_artifact(self, id: str, artifact: any) -> None:
    logging.info("Save artifact to cache...")
    artifact_file = os.path.join(self.cache_dir, id, ".json")
    with open(artifact_file, "w") as f:
        json.dump(artifact, f)


@handle_exception("Unable to retrieve artifact from cache.")
def _load_artifact(self, id: str) -> any:
    logging.info("Retrieve artifact from cache...")
    artifact_file = os.path.join(self.cache_dir, id, ".json")

    try:
        with open(artifact_file, "r") as f:
            return json.load(f)

    except Exception as e:
        logging.error(f" {artifact_file}")
        raise e


class CacheHandler:
    items = {}

    @handle_exception("Unable to instanciate CachHandler.")
    def __init__(self, run_id: str) -> None:
        self._temp_dir = tempfile.gettempdir()
        self.cache_dir = os.path.join(self._temp_dir, run_id)

        os.mkdir(self.cache_dir)
        logging.info(f"Successfully created cache directory for run {run_id}")

    def save(self, artifacts: dict[str, any]) -> None:
        for id, artifact in artifacts.items():
            _save_artifact(id, artifact)

    def retrieve(self, identifiers: list) -> dict[str, any]:
        return {id: _load_artifact(id) for id in identifiers}
