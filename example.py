import mlflow
import os
import logging
from workflow import SimpleWorkflow, Step

sampling_params = {"binary_method": "featurewise"}
learning_params = {
    "method": "rf",
    "nfp": "ResponseRate",
    "random_state": 1,
    "max_features": 4,
    "n_estimators": 15,
    "min_samples_leaf": 2,
}

CWD = os.getcwd()

workflow = SimpleWorkflow()
workflow.set_system(data_dir=os.path.join(CWD, "data/Apache"))
workflow.set_sampling(sampling_params)
workflow.set_learning(learning_params)
workflow.execute()


# system_id = mlflow.run(
#     ".",
#     entry_point="systems",
#     parameters={"param_file": "run.yaml"},
#     experiment_id=4,
# ).run_id

# sampling_id = mlflow.run(
#     "splc-sampling/",
#     parameters={
#         "system_run_id": system_id,
#         "binary_method": "featurewise",
#         "numeric_method": "centralcomposite",
#     },
#     experiment_id=1,
# ).run_id

# learning_id = mlflow.run(
#     ".",
#     entry_point="learning",
#     parameters={"sampling_run_id": sampling_id, "method": "rf", "nfp": "Time"},
#     experiment_id=2,
# ).run_id

# evaluation_id = mlflow.run(
#     ".",
#     entry_point="evaluation",
#     parameters={"learning_run_id": learning_id},
#     experiment_id=3,
# ).run_id
