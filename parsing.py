import yaml
import copy
import logging
import itertools
from rich.logging import RichHandler
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import mlflow
from experiment import SimpleExperiment, MultiStepExperiment
from steps import StepFactory

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


def _substitute_params(parameters, experiment):
    substitutions = _expand_params(parameters) if parameters else []
    experiments = []
    for exp_params in substitutions:
        tmp_exp = copy.deepcopy(experiment)
        for sub in exp_params:
            tmp_exp[sub[0]]["params"][sub[1]] = sub[2]
        experiments.append(tmp_exp)

    return experiments


def _load_run_infos(run_id, entrypoint):
    run = mlflow.get_run(run_id)

    data = {}
    data.update(run.info)
    data.update(run.data.tags)
    data.update({f"param.{k}": val for k, val in run.data.params.items()})
    data.update({f"metric.{k}": val for k, val in run.data.metrics.items()})

    data = {f"{entrypoint}.{cname}": value for cname, value in data.items()}

    return data


def _load_data(run_ids, repetition):
    data = _load_run_infos(run_ids["system_run_id"], "system")
    data.update(_load_run_infos(run_ids["sampling_run_id"], "sampling"))
    data.update(_load_run_infos(run_ids["learning_run_id"], "learning"))
    data["repetition"] = repetition

    return data


def _exp_from_config(config):
    exp = SimpleExperiment()
    exp.set_system(config["system"]["source"], config["system"]["params"])
    exp.set_sampling(config["sampling"]["source"], config["sampling"]["params"])
    exp.set_learning(config["learning"]["source"], config["learning"]["params"])
    exp.set_evaluation(config["evaluation"]["source"])
    return exp


def _generate_simple_experiments(parameters, experiment):
    exp_confs = _substitute_params(parameters, experiment)

    exps = []
    for conf in exp_confs:
        exp = _exp_from_config(conf)
        exps.append(exp)

    # with ThreadPoolExecutor(max_workers=threads) as executor:
    #     exps = executor.map(_exp_from_config, exp_confs)
    # return [e for e in exps]
    return exps


def _set_logging_to_file(config):
    for step in config:
        config[step]["params"]["logs_to_artifact"] = True

    return config


class Executor:
    def __init__(self, config_file):
        self.experiments = {}
        self.run_ids = {}
        self.data = []
        self.config, self.experiment = self._load_config(config_file)
        self._log_run_information()

    def _log_run_information(self):
        n_runs = self.config["repetitions"]
        for step in self.config["parametrization"]:
            for param in self.config["parametrization"][step]:
                n_runs = n_runs * len(self.config["parametrization"][step][param])

        logging.info(
            "Execution will lead to %i runs in %i threads.",
            n_runs,
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
            self.experiments[r] = _generate_simple_experiments(
                self.config["parametrization"],
                self.experiment,
            )

    def _execute_system_loading(self):
        system = StepFactory(
            self.experiment["system"]["source"],
            self.experiment["system"]["params"],
        )
        system.run()

    def _execute_experiments(self):
        for r, exps in self.experiments.items():
            logging.info("Start repetition %i of %i", r, self.config["repetitions"])

            with ThreadPoolExecutor(max_workers=self.config["threads"]) as executor:
                run_ids = executor.map(lambda exp: exp.execute(), exps)

            self.run_ids[r] = [x for x in run_ids]

    def execute(self):
        self._execute_system_loading()
        self._load_experiments()
        self._execute_experiments()

    def get_csv(self):
        for repetition, runs in self.run_ids.items():
            for run in runs:
                self.data.append(_load_data(run, repetition))
        return self.data
