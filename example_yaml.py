import pandas as pd
from parsing import Executor


exp = Executor("experiment.yaml")
exp.execute()
full = exp.full_data
aggregated = exp.experiment_data
pd.DataFrame(full).to_csv("test-full.csv", index=False)
pd.DataFrame(aggregated).to_csv("test-aggr.csv", index=False)
