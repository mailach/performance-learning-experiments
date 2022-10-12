from z3 import Solver, Or, Bool, Not


class FeatureModel():

    def _make_literal(self, literal: str):
        if literal[0] == "-":
            return(Not(Bool(literal[1:])))
        else:
            return(Bool(literal))

    def _solver_add_clauses(self):
        for id, name in self.features.items():
            self.solver.add(Or(self._make_literal(
                "-" + id), self._make_literal(id)))

    def _solver_add_features(self):
        for constraint in self.constraints:
            self.solver.add(Or(*[self._make_literal(literal)
                            for literal in constraint]))

    def _generate_solver(self):
        self.solver = Solver()

        self._solver_add_features()
        self._solver_add_clauses()

    def from_dimacs(self, dimacs: list):
        self.features = {line.split()[1]: line.split()[2]
                         for line in dimacs if line[0] == "c"}

        self.constraints = [line.split()
                            for line in dimacs if line[0] not in ["p", "c"]]

        self._generate_solver()

    def is_model(model):
        pass
