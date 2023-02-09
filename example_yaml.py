import pandas as pd
from executor.parsing import Executor


exp = Executor("configs/examples/exmpl_simple.yaml")
exp.execute()
