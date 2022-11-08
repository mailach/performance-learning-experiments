import mlflow
import logging


system_id = mlflow.run(
    ".",
    entry_point="systems",
    parameters={"param_file": "run.yaml"},
    experiment_id=4,
).run_id

sampling_id = mlflow.run(
    "splc-sampling/",
    parameters={
        "system_run_id": system_id,
        "binary_method": "featurewise",
        "numeric_method": "centralcomposite",
    },
    experiment_id=1,
).run_id

learning_id = mlflow.run(
    ".",
    entry_point="learning",
    parameters={"sampling_run_id": sampling_id, "method": "rf", "nfp": "Time"},
    experiment_id=2,
).run_id

evaluation_id = mlflow.run(
    ".",
    entry_point="evaluation",
    parameters={"learning_run_id": learning_id},
    experiment_id=3,
).run_id
