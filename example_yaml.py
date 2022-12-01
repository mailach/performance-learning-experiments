import pandas as pd
from parsing import Executor


exp = Executor("experiment.yaml")
exp.execute()
data = pd.DataFrame(exp.get_csv())
data.to_csv("test.csv", index=False)
