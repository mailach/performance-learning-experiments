import os

from executor.experiment import MultiStepExperiment

CWD = os.getcwd()

system_params = {"data_dir": os.path.join(CWD, "executor/resources/data/Apache")}
learning_params = {"method": "rf", "nfp": "ResponseRate"}


exp = MultiStepExperiment("multistep_example")
exp.set_system("systems", system_params)
exp.set_multistep(
    "sampling",
    [
        ("splc-sampling", {"binary_method": "featurewise"}),
        ("splc-sampling", {"binary_method": "pairwise"}),
    ],
)
exp.set_multistep(
    "learning",
    [
        ("sklearn-learning", learning_params),
        ("sklearn-learning", {"method": "rf", "nfp": "ResponseRate"}),
    ],
)
exp.execute()
