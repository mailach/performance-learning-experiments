import os

from experiment import SimpleExperiment

from steps import CustomStep


CWD = os.getcwd()

system_params = {"data_dir": os.path.join(CWD, "resources/data/Apache")}
sampling_params = {"binary_method": "featurewise"}


custom_learner_step = CustomStep(
    "https://github.com/mailach/deepperf-mlflow.git",
    "learning",
    {"nfp": "ResponseRate"},
)


exp = SimpleExperiment()
exp.set_system("systems", system_params)
exp.set_sampling("splc-sampling", sampling_params)
exp.set_learning(custom=custom_learner_step)
exp.execute()
