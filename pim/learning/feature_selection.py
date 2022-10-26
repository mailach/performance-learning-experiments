from abc import ABC, abstractmethod
import logging

import pandas as pd

from pim.learning.metrics import prediction_fault_rate


class FeatureSelector(ABC):
    @abstractmethod
    def select(
        self,
        candidates: pd.DataFrame,
        Y: pd.Series,
        feature_set: pd.DataFrame = pd.DataFrame(),
        error: float = float("inf"),
    ):
        pass


class ForwardFeatureSelector(FeatureSelector):
    def __init__(self, model):
        self.model = model

    def _find_best(
        self,
        candidates: pd.DataFrame,
        Y: pd.Series,
        feature_set: pd.DataFrame = pd.DataFrame(),
        error: float = float("inf"),
    ):
        best = None
        for candidate in candidates:
            test_set = feature_set.copy()
            test_set[candidate] = candidates[candidate]
            self.model.fit(test_set, Y)
            current_error = prediction_fault_rate(self.model.predict(test_set), Y)[
                "mean_fault_rate"
            ]

            if current_error < error:
                error = current_error
                best = candidate
        return best, error

    def select(
        self,
        candidates: pd.DataFrame,
        Y: pd.Series,
        feature_set: pd.DataFrame = pd.DataFrame(),
        error: float = float("inf"),
    ):

        for _ in candidates.columns:
            best, error = self._find_best(candidates, Y, feature_set, error)
            if best:
                feature_set[best] = candidates[best]
                candidates.drop(best, axis=1, inplace=True)

                logging.info(
                    "Add %s to featureset. Current error: %f. ",
                    best,
                    round(error, ndigits=4),
                )
            else:
                logging.info("No new suitable candidates found.")
                break

        return feature_set, error


class BackwardFeatureSelector(FeatureSelector):
    def __init__(self, model):
        raise NotImplementedError


def SelectionFactory(direction, model):
    selectors = {"forward": ForwardFeatureSelector, "backward": BackwardFeatureSelector}

    return selectors[direction](model)
