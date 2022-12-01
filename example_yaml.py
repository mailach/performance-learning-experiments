from parsing import parse_workflow


exp = parse_workflow("experiment.yaml")
exp.execute()
