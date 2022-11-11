import yaml
import logging
import rich
from rich.logging import RichHandler
import sys
from collections import Counter

from workflow import MultiStepWorkflow

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

    workflow = MultiStepWorkflow()
    workflow.set_system(system["source"], system["params"])
    workflow.set_multistep("sampling", [tuple(s.values()) for s in sampling])
    workflow.set_multistep("learning", [tuple(s.values()) for s in learning])
    workflow.set_evaluation(**evaluation)

    return workflow


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
