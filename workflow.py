import sys
import logging
from rich.logging import RichHandler
import mlflow
from steps import Step, StepFactory


logging.basicConfig(
    level=logging.INFO,
    format="WORKFLOW    %(message)s",
    handlers=[RichHandler()],
)

log = logging.getLogger("rich")


class SimpleWorkflow:

    system: Step = None
    sampling: Step = None
    learning: Step = None
    evaluation: Step = None

    def __init__(self):
        self.evaluation = StepFactory("evaluation")

    def set_system(self, source, params: dict = None):
        """set the system used in this workflow"""
        self.system = StepFactory(source, params)

    def set_sampling(self, source=None, params: dict = None, custom: Step = None):
        """specify sampling step"""
        self.sampling = custom if custom else StepFactory(source, params)

    def set_learning(self, source=None, params: dict = None, custom: Step = None):
        """specify learning step"""
        self.learning = custom if custom else StepFactory(source, params)

    def set_evaluation(self, source=None, params: dict = None, custom: Step = None):
        """specify learning step"""
        self.evaluation = custom if custom else StepFactory(source, params)

    def execute(self, backend=None, backend_config=None):
        """execute specified steps"""

        if None in [self.system, self.sampling, self.learning, self.evaluation]:
            log.error("Specify all steps prior to execution. Exiting...")
            sys.exit()
        with mlflow.start_run() as run:
            ids = {}

            ids["workflow_run_id"] = run.info.run_id
            ids["system_run_id"] = self.system.run()

            self.sampling.params["system_run_id"] = ids["system_run_id"]
            ids["sampling_run_id"] = self.sampling.run()

            self.learning.params["sampling_run_id"] = ids["sampling_run_id"]
            ids["learning_run_id"] = self.learning.run()

            self.evaluation.params["learning_run_id"] = ids["learning_run_id"]
            ids["evaluation_run_id"] = self.evaluation.run()

        if backend and backend_config:
            pass

        return ids


class MultiLearnerWorkflow:
    pass


class MultiSamplerWorkflow:
    pass
