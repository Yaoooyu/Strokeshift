"""Center-wise Global Modality Selection (CGMS)."""

from dataclasses import dataclass

import numpy as np

from src.models.calibration import PlattCalibrator
from src.models.stacking import ProbabilityStacker
from src.utils.metrics import brier_score


MODALITY_INDEX = {"CT": 0, "EHR": 1, "Report": 2}


def strategy_modalities(strategy: str) -> tuple[str, ...]:
    """Expand a candidate strategy name into calibrated modalities."""
    if strategy == "none":
        return ()
    if strategy == "All":
        return tuple(MODALITY_INDEX)
    names = tuple(strategy.split("+"))
    unknown = set(names).difference(MODALITY_INDEX)
    if unknown:
        raise ValueError(f"Unknown modalities in strategy {strategy!r}: {sorted(unknown)}")
    return names


@dataclass
class CandidateResult:
    """Fitted calibration candidate and its support-set score."""

    strategy: str
    calibrators: dict[str, PlattCalibrator]
    support_score: float


class CGMSStacking:
    """Candidate calibration bank, center-wise selection, and final prediction.

    The stacker must be fitted without target query labels. Candidate
    calibrators use support labels only. A lower support Brier score is used by
    the public reference selector; the paper configuration can substitute a
    pre-registered selection rule through this class boundary.
    """

    def __init__(
        self,
        stacker: ProbabilityStacker,
        candidate_strategies: list[str],
        random_state: int = 42,
    ) -> None:
        self.stacker = stacker
        self.candidate_strategies = candidate_strategies
        self.random_state = random_state
        self.candidates: list[CandidateResult] = []
        self.selected: CandidateResult | None = None

    def _fit_candidate(
        self,
        strategy: str,
        support_probabilities: np.ndarray,
        support_labels: np.ndarray,
    ) -> CandidateResult:
        transformed = np.asarray(support_probabilities, dtype=float).copy()
        calibrators: dict[str, PlattCalibrator] = {}
        for modality in strategy_modalities(strategy):
            index = MODALITY_INDEX[modality]
            calibrator = PlattCalibrator(self.random_state).fit(
                transformed[:, index], support_labels
            )
            transformed[:, index] = calibrator.transform(transformed[:, index])
            calibrators[modality] = calibrator
        predictions = self.stacker.predict_proba(transformed)
        return CandidateResult(strategy, calibrators, brier_score(support_labels, predictions))

    def fit(self, support_probabilities: np.ndarray, support_labels: np.ndarray) -> "CGMSStacking":
        """Fit all candidates and select one strategy for the target center."""
        self.candidates = [
            self._fit_candidate(strategy, support_probabilities, support_labels)
            for strategy in self.candidate_strategies
        ]
        self.selected = min(self.candidates, key=lambda candidate: candidate.support_score)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply the selected center-wise calibration strategy."""
        if self.selected is None:
            raise RuntimeError("Call fit before transform or predict_proba.")
        transformed = np.asarray(probabilities, dtype=float).copy()
        for modality, calibrator in self.selected.calibrators.items():
            index = MODALITY_INDEX[modality]
            transformed[:, index] = calibrator.transform(transformed[:, index])
        return transformed

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """Return final stacked predictions after selected calibration."""
        return self.stacker.predict_proba(self.transform(probabilities))

    def candidate_scores(self) -> dict[str, float]:
        """Return auditable support scores for every candidate strategy."""
        return {candidate.strategy: candidate.support_score for candidate in self.candidates}
