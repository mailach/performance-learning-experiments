from parsing import parse_workflow


workflow = parse_workflow("workflow.yaml")
workflow.execute()
