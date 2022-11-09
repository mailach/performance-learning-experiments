import os

from workflow import SimpleWorkflow
from steps import CustomStep


# Registered sources:
# systems:
#     - "systems" -> params needs accessible data_dir
#     - "existing" -> params needs to contain "run_id"
# sampling:
#     - "sampling"
#     - "splc-sampling"
# learning:
#     - "sk-learning" #supported: RandamForest, CART, KNNR, KRR,
# evaluation:
#     - "evaluation"
CWD = os.getcwd()

system_params = {"data_dir": os.path.join(CWD, "data/Apache")}
sampling_params = {"binary_method": "featurewise"}
learning_params = {
    "method": "rf",
    "nfp": "ResponseRate",
    "random_state": 1,
    "max_features": 4,
    "n_estimators": 15,
    "min_samples_leaf": 2,
}

# custom Step(custom["path"], custom["entrypoint"], params)

custom_learner_step = CustomStep(
    "https://github.com/mailach/deepperf-mlflow.git", "main"
)


workflow = SimpleWorkflow()
workflow.set_system("systems", system_params)
workflow.set_sampling("splc-sampling", sampling_params)
# workflow.set_learning("sk-learning", learning_params)  #

print(custom_learner_step.path)
workflow.set_learning(custom=custom_learner_step)
workflow.execute()

# %%
