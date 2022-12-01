import os

from experiment import MultiStepExperiment

CWD = os.getcwd()

system_params = {"data_dir": os.path.join(CWD, "resources/data/Apache")}
learning_params = {"method": "rf", "nfp": "ResponseRate"}


exp = MultiStepExperiment()
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
        ("sk-learning", learning_params),
        ("sk-learning", {"method": "rf", "nfp": "ResponseRate"}),
    ],
)
exp.execute()
