import logging

from z3 import Solver, Or, Bool, Not, sat, And, Extract, BitVec


class ConfigurationSolver():
    def __init__(self, constraints, features):
        self.solver = Solver()
        self._constraints = constraints
        self._solver_add_constraints()
        self.literals = [Bool(l) for l in features]
        self.bitvec = self._constraints_to_bitvec()

    def _reset(self):
        self.solver = Solver()
        self._solver_add_constraints()

    def _constraints_to_bitvec(self):
        return [Or(self._clause_to_bitvec(clause))
                for clause in self._constraints]

    def _clause_to_bitvec(self, clause: list):
        bitvec_size = BitVec("size",  len(self.literals) + 1)
        bitvec_rep = []
        for option in clause:
            option = int(option)
            enabled = 1 if option > 0 else 0
            bitvec_rep.append(Extract(abs(option), abs(
                option), bitvec_size) == enabled)
        return bitvec_rep

    def _make_literal(self, literal: str):
        if literal[0] == "-":
            return(Not(Bool(literal[1:])))
        else:
            return(Bool(literal))

    def _solver_add_constraints(self):
        for constraint in self._constraints:
            self.solver.add(Or(*[self._make_literal(literal)
                            for literal in constraint]))

    def generate_configurations(self, n: int) -> list():
        confs = []
        while self.solver.check() == sat and len(confs) < n:
            model = self.solver.model()
            conf = {str(l): 0 for l in self.literals}
            for l in self.literals:
                if model.evaluate(l, model_completion=True):
                    conf[str(l)] = 1
            confs.append(conf)
            self.solver.add(
                Or([p != v for p, v in [(v, model.evaluate(v, model_completion=True)) for v in self.literals]]))

        self._reset()
        if len(confs) < n:
            logging.warning(
                f"Only {len(confs)} valid configurations where created instead of {n} requested.")
        return confs

    def is_valid(self, config: dict[str: int]) -> bool:
        c = [Bool(option) for option, ind in config.items() if ind == 1]
        c += [Not(Bool(option))for option, ind in config.items() if ind == 0]
        valid = self.solver.check(*c)

        return True if valid == sat else False

    def get_minimal(self):
        min = [c[0] for c in self._constraints if len(c) == 1 and c[0] != "-"]
        config = {str(l): 1 if str(l) in min else 0 for l in self.literals}
        return config

    def mandatory(self, option: str):
        mandatory = [c[0]
                     for c in self._constraints if len(c) == 1]
        return True if option in mandatory else False


class FeatureModel():

    def __init__(self, dimacs=None, xml=None):
        if dimacs:
            self._from_dimacs(dimacs)
        elif xml:
            self._from_xml(xml)
        else:
            logging.error("You need to provider either dimacs or xml data")
            raise Exception()

    def _from_dimacs(self, dimacs: list) -> None:
        self.features = {line.split()[1]: line.split()[2]
                         for line in dimacs if line[0] == "c"}

        self.constraints = []

        self.constraints = [line.replace(" 0", "").split()
                            for line in dimacs if line[0] not in ["p", "c"]]

    def _from_xml(self, xml):
        raise NotImplementedError

    def is_model(model):
        pass
