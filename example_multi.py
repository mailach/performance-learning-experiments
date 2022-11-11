import os

from workflow import MultiStepWorkflow

CWD = os.getcwd()

system_params = {"data_dir": os.path.join(CWD, "data/Apache")}
sampling_params = {"binary_method": "featurewise"}
learning_params = {"method": "rf", "nfp": "ResponseRate"}


workflow = MultiStepWorkflow()
workflow.set_system("systems", system_params)
workflow.set_multistep("sampling", [("splc-sampling", sampling_params)])
workflow.set_multistep("learning", [("sk-learning", learning_params)])
workflow.execute()
