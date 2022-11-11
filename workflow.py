import logging
import sys
from abc import ABC
from concurrent.futures import ThreadPoolExecutor

import mlflow
from rich.logging import RichHandler

from steps import Step, StepFactory

logging.basicConfig(
    level=logging.INFO,
    format="WORKFLOW    %(message)s",
    handlers=[RichHandler()],
)

log = logging.getLogger("rich")


class Workflow(ABC):
    steps: dict = {
        "system": None,
        "sampling": None,
        "learning": None,
        "evaluation": None,
    }

    def __init__(self):
        self.steps["evaluation"] = StepFactory("evaluation")

    def _all_steps_set_or_exit(self):
        if None in self.steps.values():
            log.error("Specify all steps prior to execution. Exiting...")
            sys.exit()

    def set_system(self, source, params: dict = None):
        """set the system used in this workflow"""
        self.steps["system"] = StepFactory(source, params)

    def set_evaluation(self, source=None, params: dict = None, custom: Step = None):
        """specify learning step"""
        self.steps["evaluation"] = custom if custom else StepFactory(source, params)


class SimpleWorkflow(Workflow):
    def set_sampling(self, source=None, params: dict = None, custom: Step = None):
        """specify sampling step"""
        self.steps["sampling"] = custom if custom else StepFactory(source, params)

    def set_learning(self, source=None, params: dict = None, custom: Step = None):
        """specify learning step"""
        self.steps["learning"] = custom if custom else StepFactory(source, params)

    def execute(self, backend=None, backend_config=None):
        """execute specified steps"""

        self._all_steps_set_or_exit()

        with mlflow.start_run() as run:
            ids = {}

            ids["workflow_run_id"] = run.info.run_id
            ids["system_run_id"] = self.steps["system"].run()

            self.steps["sampling"].params["system_run_id"] = ids["system_run_id"]
            ids["sampling_run_id"] = self.steps["sampling"].run()

            self.steps["learning"].params["sampling_run_id"] = ids["sampling_run_id"]
            ids["learning_run_id"] = self.steps["learning"].run()

            self.steps["evaluation"].params["learning_run_id"] = ids["learning_run_id"]
            ids["evaluation_run_id"] = self.steps["evaluation"].run()

        if backend and backend_config:
            pass

        return ids


class MultiStepWorkflow(Workflow):
    def _generate_step_list(self, steps: list):
        return [StepFactory(step_info[0], step_info[1]) for step_info in steps]

    def _check_stepname(self, step_name):
        if step_name not in self.steps:
            logging.error(
                "%s is not a valid step_name. Valid steps are %s",
                step_name,
                ", ".join(self.steps.keys()),
            )

    def set_multistep(self, step_name, steps: list = None, custom: list = None):

        self._check_stepname(step_name)
        self.steps[step_name] = self._generate_step_list(steps) if steps else []

        self.steps[step_name] += self._generate_step_list(custom) if custom else []

    def _warn_if_no_multistep(self):
        if len(self.steps["learning"]) == 1 and len(self.steps["sampling"]) == 1:
            logging.warning(
                "Running a simple workflow as multistep. Consider using SimpleWorkflow class."
            )

    def _execute_multiple_steps(self, step_name, threads=5):

        with ThreadPoolExecutor(max_workers=threads) as executor:
            run_ids = executor.map(lambda step: step.run(), self.steps[step_name])

        return [idx for idx in run_ids]

    def _generate_sampling_steps(self, system_run_id):

        for step in self.steps["sampling"]:
            step.params["system_run_id"] = system_run_id

    def _generate_evaluation_steps(self, learning_runs):
        self.steps["evaluation"] = []
        for idx in learning_runs:
            step = StepFactory("evaluation")
            step.params["learning_run_id"] = idx
            self.steps["evaluation"].append(step)

    def _generate_learning_steps(self, sampling_runs):
        new_steps = []

        for step in self.steps["learning"]:
            for idx in sampling_runs:
                new_step = step.deepcopy()
                new_step.params["sampling_run_id"] = idx
                new_steps.append(new_step)

        self.steps["learning"] = new_steps

    def execute(self, backend=None, backend_config=None):
        """execute specified steps"""

        self._all_steps_set_or_exit()

        self._warn_if_no_multistep()

        with mlflow.start_run() as run:
            ids = {}

            # run system run
            ids["workflow"] = run.info.run_id
            ids["system"] = self.steps["system"].run()

            # run first possible multistep workflow
            self._generate_sampling_steps(ids["system"])
            ids["sampling"] = self._execute_multiple_steps("sampling")

            self._generate_learning_steps(ids["sampling"])
            ids["learning"] = self._execute_multiple_steps("learning")

            self._generate_evaluation_steps(ids["learning"])
            ids["evaluation"] = self._execute_multiple_steps("evaluation")

        if backend and backend_config:
            pass

        return ids
