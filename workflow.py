import mlflow


class Step:
    run_id = None

    def __init__(self, path: str, entry_point: str, params: dict = None, run_id=None):
        self.path = path
        self.entry_point = entry_point
        self.params = params if params else {}
        self.run_id = run_id

    @classmethod
    def from_run_id(cls, run_id):
        """create step object from an existing run_id"""
        return cls(None, None, None, run_id)

    def run(self):
        """either runs specified project or returns existing run"""
        if not self.run_id:
            self.run_id = mlflow.run(
                self.path,
                entry_point=self.entry_point,
                parameters=self.params,
                experiment_name=self.entry_point,
            ).run_id
        return self.run_id


class SimpleWorkflow:

    system: Step = None
    sampling: Step = None
    learning: Step = None
    evaluation: Step = None

    def __init__(self):
        self.evaluation = Step("steps/", "evaluation")

    def set_system(self, system_run_id: str = None, data_dir: str = None):
        """set the system used in this workflow"""
        if system_run_id:
            self.system = Step.from_run_id(system_run_id)
        elif data_dir:
            self.system = Step("steps/", "systems", {"data_dir": data_dir})

    def set_sampling(self, params, custom: bool = False):
        """specify sampling step"""
        if not custom:
            if "true_random" not in params:
                self.sampling = Step("steps/splc-sampling/", "sampling", params)
            else:
                self.sampling = Step("steps/", "sampling", params)

    def set_learning(self, params, custom: bool = False):
        """specify learning step"""
        if not custom:
            self.learning = Step("steps/", "learning", params)

    def execute(self, backend=None, backend_config=None):
        with mlflow.start_run() as run:

            workflow_run_id = run.info.run_id
            system_run_id = self.system.run()

            self.sampling.params["system_run_id"] = system_run_id
            sampling_run_id = self.sampling.run()

            self.learning.params["sampling_run_id"] = sampling_run_id
            learning_run_id = self.learning.run()

            self.evaluation.params["learning_run_id"] = learning_run_id
            evaluation_run_id = self.evaluation.run()

        if backend and backend_config:
            pass

        return {
            "workflow_run_id": workflow_run_id,
            "system_run_id": system_run_id,
            "sampling_run_id": sampling_run_id,
            "learning_run_id": learning_run_id,
            "evaluation_run_id": evaluation_run_id,
        }


class MultiLearnerWorkflow:
    pass


class MultiSamplerWorkflow:
    pass
