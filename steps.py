import mlflow


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
        self.run_id = mlflow.run(
            self.path,
            entry_point=self.entry_point,
            parameters=self.params,
            experiment_name=self.entry_point,
        ).run_id

        return self.run_id


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
        self.path = "steps/"
        self.entry_point = "systems"
        self.params = params if params else {}


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
