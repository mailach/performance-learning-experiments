import os

from experiment import SimpleExperiment

CWD = os.getcwd()


system_params = {"data_dir": os.path.join(CWD, "resources/data/Apache")}
sampling_params = {"method": "random", "n": 10}
learning_params = {
    "resampleMethod": "crossvalidation",
    "nfp": "ResponseRate",
    "paraSearchMethod": "randomsearch",
}


exp = SimpleExperiment(
    "simple_example",
    backend="kubernetes",
    backend_config="resources/config/k8s/k8s_config.json",
)
exp.set_system("systems", system_params)
exp.set_sampling("sklearn-sampling", sampling_params)
exp.set_learning("decart", learning_params)
exp.execute()
