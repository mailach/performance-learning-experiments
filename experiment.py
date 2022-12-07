import logging
import sys
from abc import ABC
from concurrent.futures import ThreadPoolExecutor

import mlflow
from mlflow.tracking import MlflowClient

from rich.logging import RichHandler

from steps import Step, StepFactory

logging.basicConfig(
    level=logging.INFO,
    format="EXPERIMENT    %(message)s",
    handlers=[RichHandler()],
)


def _load_params_and_metrics(run_id):
    run = mlflow.get_run(run_id)
    return run.data.params, run.data.metrics


def _update_run_data(experiment_id, sub_run_id, sub_run_name, client):
    params, metrics = _load_params_and_metrics(sub_run_id)
    params = {f"{sub_run_name}.{k}": v for k, v in params.items()}
    metrics = {f"{sub_run_name}.{k}": v for k, v in metrics.items()}

    for k, v in metrics.items():
        client.log_metric(experiment_id, k, v)
    for k, v in params.items():
        client.log_param(experiment_id, k, v)


def _update_exp_params_and_metrics(ids, client):
    steps = [step.replace("run_id", "") for step in ids if step != "experiment"]
    for step in steps:
        _update_run_data(ids["experiment"], ids[step], step, client)


class Experiment(ABC):
    def __init__(self, experiment_name: str = None):
        self.steps = {
            "system": None,
            "sampling": None,
            "learning": None,
            "evaluation": None,
        }
        self.experiment_name = experiment_name
        self.steps["evaluation"] = StepFactory("evaluation")
        self.client = MlflowClient()

    def _all_steps_set_or_exit(self):
        if None in self.steps.values():
            logging.error("Specify all steps prior to execution. Exiting...")
            sys.exit()

    def set_system(self, source, params: dict = None):
        """set the system used in this workflow"""
        self.steps["system"] = StepFactory(source, params)

    def set_evaluation(self, source=None, params: dict = None, custom: Step = None):
        """specify learning step"""
        self.steps["evaluation"] = custom if custom else StepFactory(source, params)


class SimpleExperiment(Experiment):
    def set_sampling(self, source=None, params: dict = None, custom: Step = None):
        """specify sampling step"""
        self.steps["sampling"] = custom if custom else StepFactory(source, params)

    def set_learning(self, source=None, params: dict = None, custom: Step = None):
        """specify learning step"""
        self.steps["learning"] = custom if custom else StepFactory(source, params)

    def execute(self, backend=None, backend_config=None):
        """execute specified steps"""

        self._all_steps_set_or_exit()
        mlflow.set_experiment(experiment_name=self.experiment_name)

        experiment = self.client.search_experiments(
            filter_string=f"name = '{self.experiment_name}'"
        )
        if len(experiment):
            exp_id = experiment[0].experiment_id
        else:
            exp_id = self.client.create_experiment(self.experiment_name)

        run = self.client.create_run(exp_id)
        ids = {}
        ids["experiment"] = run.info.run_id

        ids["system"] = self.steps["system"].run()

        self.steps["sampling"].params["system_run_id"] = ids["system"]
        ids["sampling"] = self.steps["sampling"].run()

        self.steps["learning"].params["sampling_run_id"] = ids["sampling"]
        ids["learning"] = self.steps["learning"].run()

        self.steps["evaluation"].params["learning_run_id"] = ids["learning"]
        ids["evaluation"] = self.steps["evaluation"].run()

        self.client.set_terminated(run.info.run_id)
        _update_exp_params_and_metrics(ids, self.client)

        return ids


class MultiStepExperiment(Experiment):
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

    def _execute_multiple_steps(self, step_name, threads=2):

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

        experiment = self.client.search_experiments(
            filter_string=f"name = '{self.experiment_name}'"
        )
        if len(experiment):
            exp_id = experiment[0].experiment_id
        else:
            exp_id = self.client.create_experiment(self.experiment_name)

        run = self.client.create_run(exp_id)
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
        self.client.set_terminated(run.info.run_id)

        return ids
