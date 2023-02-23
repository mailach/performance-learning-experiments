from distutils.util import execute
import yaml
import copy
import logging
import itertools
from rich.logging import RichHandler
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import mlflow
from executor.experiment import SimpleExperiment, MultiStepExperiment
from executor.steps import StepFactory

logging.basicConfig(
    level=logging.INFO,
    format="WORKFLOW    %(message)s",
    handlers=[RichHandler()],
)

STEPS = ["system", "sampling", "learning", "evaluation"]


def _extract_steps(content):
    return (content[step] for step in STEPS)


def parse_workflow(filename):

    with open(filename, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)

    _validate(content)

    system, sampling, learning, evaluation = _extract_steps(content)

    exp = MultiStepExperiment()
    exp.set_system(system["source"], system["params"])
    exp.set_multistep("sampling", [tuple(s.values()) for s in sampling])
    exp.set_multistep("learning", [tuple(s.values()) for s in learning])
    exp.set_evaluation(**evaluation)


def _validate_system(system):
    if not isinstance(system, dict):
        logging.error("Your yamlfile has no object for step system. Exiting...")
        sys.exit(1)


def _validate_steps(steps):
    if Counter(steps) != Counter(STEPS):
        logging.error("Your yamlfile needs to contain %s. Exiting...", ", ".join(STEPS))
        sys.exit(1)


def _validate(content: dict):

    _validate_steps(content.keys())
    _validate_system(content["system"])


#########################################################
def _params_to_list(params):
    param_list = []
    param_names = []
    for step in params:
        for param_name, param in params[step].items():
            param_list.append(param)
            param_names.append((step, param_name))

    return param_list, param_names


def _generate_param_substitutions(params, names):
    substitutions = []
    params = [prod for prod in itertools.product(*params)]
    for param_tup in params:
        conf = []
        for i, name in enumerate(names):
            conf.append((name[0], name[1], param_tup[i]))
        substitutions.append(conf)
    return substitutions


def _expand_params(params):
    params, names = _params_to_list(params)
    subs = _generate_param_substitutions(params, names)
    return subs


def _substitute_crossprod_params(parameters, experiment):
    substitutions = _expand_params(parameters) if parameters else []
    experiments = []
    for exp_params in substitutions:
        tmp_exp = copy.deepcopy(experiment)
        for sub in exp_params:
            tmp_exp[sub[0]]["params"][sub[1]] = sub[2]
        experiments.append(tmp_exp)

    return experiments


def _substitute_stepwise_params(parameters, experiment):
    stepname = parameters["stepname"]

    experiments = []
    for parameters in parameters["params"]:
        tmp_exp = copy.deepcopy(experiment)
        for p, v in parameters.items():
            tmp_exp[stepname]["params"][p] = v
        experiments.append(tmp_exp)

    return experiments

def _load_run_infos(run_id, entrypoint=None):
    run = mlflow.get_run(run_id)

    data = {}
    data.update(run.info)
    data.update(run.data.tags)
    if entrypoint:
        data.update({f"param.{k}": val for k, val in run.data.params.items()})
        data.update({f"metric.{k}": val for k, val in run.data.metrics.items()})
        data = {f"{entrypoint}.{cname}": value for cname, value in data.items()}
    else:
        data.update(run.data.params)
        data.update(run.data.metrics)

    return data


def _load_data(run_ids, repetition):
    full_data = _load_run_infos(run_ids["system"], "system")
    full_data.update(_load_run_infos(run_ids["sampling"], "sampling"))
    full_data.update(_load_run_infos(run_ids["learning"], "learning"))
    full_data["repetition"] = repetition

    aggregated_data = _load_run_infos(run_ids["experiment"])

    return full_data, aggregated_data


def _exp_from_config(config, experiment_name):
    exp = SimpleExperiment(experiment_name)
    exp.set_system(config["system"]["source"], config["system"]["params"])
    exp.set_sampling(config["sampling"]["source"], config["sampling"]["params"])
    exp.set_learning(config["learning"]["source"], config["learning"]["params"])
    exp.set_evaluation(config["evaluation"]["source"])
    return exp


def _parameterize_experiments(parameters, experiment, experiment_name):
    if parameters["type"] == "crossproduct":
        exp_confs = _substitute_crossprod_params({k:v for k,v in parameters.items() if k!="type"}, experiment)
    elif parameters["type"] == "stepwise":
        exp_confs = _substitute_stepwise_params({k:v for k,v in parameters.items() if k!="type"}, experiment)
    else:
        raise NotImplementedError("Substitution type not implemented yet")

    exps = []
    for conf in exp_confs:
        exp = _exp_from_config(conf, experiment_name)
        exps.append(exp)

    return exps


def _simple_experiment(experiment, experiment_name):
    return _exp_from_config(experiment, experiment_name)

     


def _set_logging_to_file(config):
    for step in config:
        config[step]["params"]["logs_to_artifact"] = True

    return config


class Executor:
    def __init__(self, config_file):
        self.experiments = {}
        self.run_ids = {}
        self.exp_data = []
        self.sampling_data = []
        self.learning_data = []
        self.config, self.experiment = self._load_config(config_file)
        

    def _log_run_information(self):
        logging.info(
            "Execution will lead to %i runs in %i threads.",
            len(self.experiments[1]) * len(self.experiments),
            self.config["threads"],
        )

    def _load_config(self, config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)

        exp_config = content["configuration"]
        step_config = _set_logging_to_file(content["experiment"])

        return exp_config, step_config

    def _load_experiments(self):

        for r in range(1, self.config["repetitions"] + 1):
            logging.info("Generating experiments for repetition %i", r)
            if self.config["parametrization"] != "None":
                self.experiments[r] = _parameterize_experiments(
                    self.config["parametrization"], self.experiment, self.config["name"]
                )
            else:
                self.experiments[r] = [_simple_experiment(self.experiment, self.config["name"])]
                #logging.error("No parametrization not yet implemented")
                #raise NotImplementedError

    def _execute_experiments(self):
        for r, exps in self.experiments.items():
            logging.info("Start repetition %i of %i", r, self.config["repetitions"])
            executed_runs = []

            if r == 1:
                executed_runs.append(exps[0].execute())
                exps = exps[1:]

            

            if self.config["threads"] == 1:
                for e in exps:
                    executed_runs.append(e.execute())
            else:
                with ThreadPoolExecutor(max_workers=self.config["threads"]) as executor:
                    ids = executor.map(lambda exp: exp.execute(), exps)
                    executed_runs += [x for x in ids]
            

            self.run_ids[r] = [ids["experiment"] for ids in executed_runs]

            # save temporary data:
            #self._load_experiment_data()
            
    def execute(self):
        self._load_experiments()
        self._log_run_information()
        self._execute_experiments()
        #self._load_experiment_data()

    def _load_experiment_data(self):
        self.exp_data = []
        self.sampling_data = []
        self.learning_data = []
        for _, runs in self.run_ids.items():
            for run in runs:
                if "experiment" in run and "sampling" in run and "learning" in run:
                    self.exp_data.append(_load_run_infos(run["experiment"]))
                    self.sampling_data.append(_load_run_infos(run["sampling"]))
                    self.learning_data.append(_load_run_infos(run["learning"]))
    