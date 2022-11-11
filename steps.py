import os
import yaml
import logging

from rich.logging import RichHandler
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="WORKFLOW    %(message)s",
    handlers=[RichHandler()],
)


def _generate_filter_string(params: dict):
    clauses = [
        "parameter." + param + " = '" + str(value) + "'"
        for param, value in params.items()
        if param != "nfp"
    ]
    query = " AND ".join(clauses) + " AND attribute.status = 'FINISHED'"
    return query


def get_run_if_exists(entrypoint: str, parameters: dict):
    """
    Returns a run if one exists.

    Parameters
    ----------
    parameters : dict[str, any]
        the parameters that the run should contain
    """

    filter_string = _generate_filter_string(parameters)
    runs = mlflow.search_runs(
        experiment_names=[entrypoint], filter_string=filter_string
    )
    return runs["run_id"][0] if not runs.empty else False


class Step:
    path: str = None
    run_id: str = None
    entry_point: str = None
    params: dict = {}

    @classmethod
    def from_run_id(cls, run_id):
        """create step object from an existing run_id"""
        return cls(None, None, None, run_id)

    def run(self):
        """either runs specified project or returns existing run"""
        if not self.run_id:
            self.run_id = mlflow.run(
                self.path,
                entry_point=self.entry_point,
                parameters=self.params,
                experiment_name=self.entry_point,
            ).run_id
        else:
            logging.warning("Use existing system data from run %s", self.run_id)

        return self.run_id

    def deepcopy(self):
        copy = Step()
        copy.path = self.path
        copy.run_id = self.run_id
        copy.entry_point = self.entry_point
        copy.params = self.params.copy()
        return copy


class CustomStep(Step):
    def __init__(self, path: str, entry_point: str, params: dict = None, run_id=None):
        self.path = path
        self.entry_point = entry_point
        self.params = params if params else {}
        self.run_id = run_id


class NonExecutingStep(Step):
    def __init__(self, run_id):
        self.run_id = run_id

    def run(self):
        return self.run_id


class SplcSamplingStep(Step):
    def __init__(self, params: dict = None):
        self.path = "steps/splc-sampling"
        self.entry_point = "sampling"
        self.params = params if params else {}


class DefaultSamplingStep(Step):
    def __init__(self, params: dict = None):
        self.path = "steps/"
        self.entry_point = "sampling"
        self.params = params if params else {}


class ScikitLearnerStep(Step):
    def __init__(self, params: dict = None):
        self.path = "steps/"
        self.entry_point = "learning"
        self.params = params if params else {}


class DefaultEvaluationStep(Step):
    def __init__(self, params: dict = None):
        self.path = "steps/"
        self.entry_point = "evaluation"
        self.params = params if params else {}


class SystemLoadingStep(Step):
    def __init__(self, params: dict = None):
        self.run_id = get_run_if_exists("systems", self._load_meta(params["data_dir"]))
        self.path = "steps/"
        self.entry_point = "systems"
        self.params = params if params else {}

    def _load_meta(self, data_dir):
        with open(os.path.join(data_dir, "meta.yaml"), "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def StepFactory(source, params=None):
    sources = {
        "sk-learning": ScikitLearnerStep,
        "splc-sampling": SplcSamplingStep,
        "sampling": DefaultSamplingStep,
        "evaluation": DefaultEvaluationStep,
        "systems": SystemLoadingStep,
        "existing": NonExecutingStep,
    }

    return sources[source](params)
