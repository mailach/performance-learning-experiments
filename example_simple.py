import os

from workflow import SimpleWorkflow

CWD = os.getcwd()

system_params = {"data_dir": os.path.join(CWD, "resources/data/x264")}
sampling_params = {"method": "random", "n": 10}
learning_params = {
    "method": "rf",
    "nfp": "Performance",
    "random_state": 1,
    "max_features": 4,
    "n_estimators": 15,
    "min_samples_leaf": 2,
}


workflow = SimpleWorkflow()
workflow.set_system("systems", system_params)
workflow.set_sampling("sklearn-sampling", sampling_params)
workflow.set_learning("sk-learning", learning_params)
workflow.execute()
