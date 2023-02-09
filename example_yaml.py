import pandas as pd
from executor.parsing import Executor


exp = Executor("exmpl_stepwise.yaml")
exp.execute()
