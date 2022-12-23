import os
import yaml
import logging

from rich.logging import RichHandler
import mlflow
import mlflow.projects

logging.basicConfig(
    level=logging.INFO,
    format="STEP    %(message)s",
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


def get_system_if_exists(name):
    """
    Returns a run if one exists.

    Parameters
    ----------
    parameters : dict[str, any]
        the parameters that the run should contain
    """

    filter_string = f"parameter.system = '{name}' AND attribute.status = 'FINISHED'"
    runs = mlflow.search_runs(experiment_names=["systems"], filter_string=filter_string)
    return runs["run_id"][0] if not runs.empty else False


class Step:
    path: str = None
    run_id: str = None
    entry_point: str = None
    params: dict = None
    experiment_name: str = None

    @classmethod
    def from_run_id(cls, run_id):
        """create step object from an existing run_id"""
        return cls(None, None, None, run_id)

    def run(self):
        """either runs specified project or returns existing run"""

        retries = 0
        while retries < 3:
            try:
                if not self.run_id:
                    self.run_id = mlflow.projects.run(
                        self.path,
                        entry_point=self.entry_point,
                        experiment_name=self.experiment_name,
                        parameters=self.params,
                    ).run_id
                else:
                    logging.warning(
                        "Use existing run %s in entrypoint %s",
                        self.run_id,
                        self.entry_point,
                    )
                return self.run_id
            except Exception as e:
                logging.error(
                    "An exception occured during %i. retry of execution: %s",
                    retries + 1,
                    e,
                )
                retries += 1
        raise Exception("Could not execute step %s", self.entry_point)

    def deepcopy(self):
        copy = Step()
        copy.path = self.path
        copy.run_id = self.run_id
        copy.entry_point = self.entry_point
        copy.experiment_name = self.experiment_name
        copy.params = self.params.copy()
        return copy


class CustomStep(Step):
    def __init__(
        self,
        path: str,
        entry_point: str,
        params: dict = None,
        run_id=None,
        experiment_name: str = "custom",
    ):
        self.path = path
        self.entry_point = entry_point
        self.experiment_name = experiment_name
        self.params = params if params else {}
        self.run_id = run_id


class NonExecutingStep(Step):
    def __init__(self, run_id):
        self.run_id = run_id

    def run(self):
        return self.run_id


class SplcSamplingStep(Step):
    def __init__(self, params: dict = None):
        self.path = "steps/SPLConqueror"
        self.entry_point = "sampling"
        self.experiment_name = "splc-sampling"
        self.params = params if params else {}


class SplcLearningStep(Step):
    def __init__(self, params: dict = None):
        self.path = "steps/SPLConqueror"
        self.entry_point = "learning"
        self.experiment_name = "splc-learning"
        self.params = params if params else {}


class SklearnSamplingStep(Step):
    def __init__(self, params: dict = None):
        self.path = "steps/scikit-learn/"
        self.entry_point = "sampling"
        self.experiment_name = "sklearn-sampling"
        self.params = params if params else {}


class ScikitLearnerStep(Step):
    def __init__(self, params: dict = None):
        self.path = "steps/scikit-learn/"
        self.entry_point = "learning"
        self.experiment_name = "sklearn-learning"
        self.params = params if params else {}


class DecartLearnerStep(Step):
    def __init__(self, params: dict = None):
        self.path = "steps/learning/decart/decart"
        self.entry_point = "learning"
        self.experiment_name = "decart"
        self.params = params if params else {}


class DeepperfLearnerStep(Step):
    def __init__(self, params: dict = None):
        self.path = "steps/learning/deepperf"
        self.entry_point = "learning"
        self.experiment_name = "deepperf"
        self.params = params if params else {}


class DefaultEvaluationStep(Step):
    def __init__(self, params: dict = None):
        self.path = "steps/evaluation"
        self.entry_point = "evaluation"
        self.experiment_name = "evaluation"
        self.params = params if params else {}


class SystemLoadingStep(Step):
    def __init__(self, params: dict = None):
        self.run_id = get_system_if_exists(
            self._load_name(os.path.join(params["data_dir"]))
        )
        self.path = "steps/systems/"
        self.entry_point = "systems"
        self.experiment_name = "systems"
        self.params = params if params else {}

    def _load_name(self, data_dir):
        with open(os.path.join(data_dir, "meta.yaml"), "r", encoding="utf-8") as f:
            return yaml.safe_load(f)["system"]


def StepFactory(source, params=None):
    sources = {
        "sklearn-learning": ScikitLearnerStep,
        "sklearn-sampling": SklearnSamplingStep,
        "splc-sampling": SplcSamplingStep,
        "splc-learning": SplcLearningStep,
        "decart": DecartLearnerStep,
        "deepperf": DeepperfLearnerStep,
        "evaluation": DefaultEvaluationStep,
        "systems": SystemLoadingStep,
        "existing": NonExecutingStep,
    }

    return sources[source](params)
