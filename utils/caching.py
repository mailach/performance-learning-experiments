import tempfile
import os
import json
import logging


class CacheHandler:
    def __init__(self, run_id: str) -> None:
        self._temp_dir = tempfile.gettempdir()
        self.cache_dir = os.path.join(self._temp_dir, run_id)

        try:
            os.mkdir(self.cache_dir)
            logging.info(f"Successfully created cache directory for run " + run_id)
        except Exception as e:
            logging.error(f"Unable to create cache directory for run " + run_id)
            raise e

    def save(self, artifact: any, identifier: str) -> None:
        with open(os.path.join(self.cache_dir, identifier), "w") as f:
            json.dump(artifact, f)

    def retrieve(self, identifier: str) -> any:
        with open(os.path.join(self.cache_dir, identifier), "r") as f:
            artifact = json.load(identifier)
