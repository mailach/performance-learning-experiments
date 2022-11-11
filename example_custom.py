import os

from workflow import SimpleWorkflow

from steps import CustomStep


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


custom_learner_step = CustomStep(
    "https://github.com/mailach/deepperf-mlflow.git",
    "learning",
    {"nfp": "ResponseRate"},
)


workflow = SimpleWorkflow()
workflow.set_system("systems", system_params)
workflow.set_sampling("splc-sampling", sampling_params)
workflow.set_learning(custom=custom_learner_step)
workflow.execute()
