"""Central validation runner for federated learning round evaluation.

Evaluates the global model on a held-out central validation dataset after
each aggregation round. Tracks per-round metrics, detects convergence and
overfitting, and generates structured validation reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_ROUNDS_FOR_CONVERGENCE = 3
DEFAULT_CONVERGENCE_TOLERANCE = 1e-4
DEFAULT_OVERFITTING_THRESHOLD = 0.05  # 5% gap between train and val accuracy


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Evaluation outcome for a single federated training round.

    Attributes:
        job_id: The federated job being evaluated.
        round_number: The round this result corresponds to.
        loss: Cross-entropy (or task-specific) loss on the validation set.
        accuracy: Classification accuracy (0–1). None for regression tasks.
        f1_score: Macro-averaged F1 score. None if not applicable.
        auc_roc: Area under the ROC curve. None if not applicable.
        custom_metrics: Additional task-specific metrics.
        num_validation_samples: Number of samples in the holdout set.
        evaluated_at: UTC timestamp of the evaluation.
        converged: Whether convergence was detected at this round.
        overfitting_detected: Whether the overfitting heuristic triggered.
    """

    job_id: str
    round_number: int
    loss: float
    accuracy: float | None = None
    f1_score: float | None = None
    auc_roc: float | None = None
    custom_metrics: dict[str, float] = field(default_factory=dict)
    num_validation_samples: int = 0
    evaluated_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    converged: bool = False
    overfitting_detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "job_id": self.job_id,
            "round_number": self.round_number,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "custom_metrics": self.custom_metrics,
            "num_validation_samples": self.num_validation_samples,
            "evaluated_at": self.evaluated_at.isoformat(),
            "converged": self.converged,
            "overfitting_detected": self.overfitting_detected,
        }


# ---------------------------------------------------------------------------
# CentralValidationRunner
# ---------------------------------------------------------------------------


class CentralValidationRunner:
    """Evaluates the global model on a central holdout dataset each round.

    The runner maintains an in-memory history of round results for convergence
    and overfitting detection, and persists them to the provided repository.

    Args:
        validation_dataset: Object providing a numpy-compatible evaluate()
            interface. Must expose:
              - evaluate(parameters) -> dict[str, float]  (returns loss, accuracy, etc.)
              - num_samples -> int
        result_repository: Async repository for persisting ValidationResult records.
        baseline_metrics: Optional dict of pre-training baseline metric values
            to compare against (e.g., {"loss": 2.3, "accuracy": 0.1}).
        convergence_tolerance: Minimum loss improvement required to avoid
            triggering the convergence flag.
        overfitting_threshold: Maximum tolerable gap between training and
            validation accuracy before flagging overfitting.
        model_loader: Callable that accepts serialized bytes and returns
            loaded model weights (np.ndarray list). If None, raw parameter
            arrays are passed directly to the dataset evaluator.
    """

    def __init__(
        self,
        validation_dataset: Any,
        result_repository: Any,
        baseline_metrics: dict[str, float] | None = None,
        convergence_tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE,
        overfitting_threshold: float = DEFAULT_OVERFITTING_THRESHOLD,
        model_loader: Callable[[bytes], list[np.ndarray[Any, Any]]] | None = None,
    ) -> None:
        self._dataset = validation_dataset
        self._result_repo = result_repository
        self._baseline_metrics = baseline_metrics or {}
        self._convergence_tolerance = convergence_tolerance
        self._overfitting_threshold = overfitting_threshold
        self._model_loader = model_loader

        # job_id -> list[ValidationResult] (in-memory history for detection)
        self._history: dict[str, list[ValidationResult]] = {}

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    async def evaluate_round(
        self,
        job_id: str,
        round_number: int,
        parameters: list[np.ndarray[Any, Any]],
        training_metrics: dict[str, float] | None = None,
    ) -> ValidationResult:
        """Evaluate the global model after a completed aggregation round.

        Runs forward inference on the central validation dataset, records
        metrics, checks for convergence and overfitting, and persists the
        result.

        Args:
            job_id: The federated job being evaluated.
            round_number: The completed training round number.
            parameters: Aggregated global model weights.
            training_metrics: Optional dict of training-phase metrics (e.g.,
                train_loss, train_accuracy) for overfitting detection.

        Returns:
            ValidationResult with all computed metrics and detection flags.
        """
        raw_metrics: dict[str, float] = await self._run_evaluation(parameters)

        result = ValidationResult(
            job_id=job_id,
            round_number=round_number,
            loss=raw_metrics.get("loss", float("nan")),
            accuracy=raw_metrics.get("accuracy"),
            f1_score=raw_metrics.get("f1_score"),
            auc_roc=raw_metrics.get("auc_roc"),
            custom_metrics={
                k: v
                for k, v in raw_metrics.items()
                if k not in {"loss", "accuracy", "f1_score", "auc_roc"}
            },
            num_validation_samples=getattr(self._dataset, "num_samples", 0),
        )

        # Convergence and overfitting checks
        history = self._history.setdefault(job_id, [])
        result.converged = self._check_convergence(history, result.loss)
        result.overfitting_detected = self._check_overfitting(
            result, training_metrics or {}
        )

        history.append(result)

        await self._result_repo.save_validation_result(result)

        logger.info(
            "Validation round %d for job %s: loss=%.4f accuracy=%s "
            "converged=%s overfitting=%s",
            round_number,
            job_id,
            result.loss,
            f"{result.accuracy:.4f}" if result.accuracy is not None else "N/A",
            result.converged,
            result.overfitting_detected,
        )

        return result

    async def _run_evaluation(
        self, parameters: list[np.ndarray[Any, Any]]
    ) -> dict[str, float]:
        """Delegate model evaluation to the validation dataset.

        Args:
            parameters: Global model weight arrays.

        Returns:
            Dict of metric name -> scalar value.
        """
        if hasattr(self._dataset, "evaluate"):
            result = self._dataset.evaluate(parameters)
            # Support both sync and async dataset evaluators
            if hasattr(result, "__await__"):
                return await result
            return dict(result)

        raise TypeError(
            f"Validation dataset {type(self._dataset)} does not expose an evaluate() method"
        )

    # ------------------------------------------------------------------
    # Convergence detection
    # ------------------------------------------------------------------

    def _check_convergence(
        self,
        history: list[ValidationResult],
        current_loss: float,
    ) -> bool:
        """Detect convergence by comparing recent loss improvements.

        Convergence is declared if the loss improvement over the last
        MIN_ROUNDS_FOR_CONVERGENCE rounds is below convergence_tolerance.

        Args:
            history: Previous validation results for this job.
            current_loss: Loss in the current round.

        Returns:
            True if convergence is detected.
        """
        if len(history) < MIN_ROUNDS_FOR_CONVERGENCE:
            return False

        recent_losses = [r.loss for r in history[-MIN_ROUNDS_FOR_CONVERGENCE:]]
        max_recent = max(recent_losses)
        improvement = max_recent - current_loss

        return improvement < self._convergence_tolerance

    def check_early_stopping(
        self,
        job_id: str,
        patience: int = 5,
    ) -> bool:
        """Check if training should stop early due to no improvement.

        Args:
            job_id: The job to check.
            patience: Number of consecutive non-improving rounds before stopping.

        Returns:
            True if early stopping should be triggered.
        """
        history = self._history.get(job_id, [])
        if len(history) < patience:
            return False

        recent = history[-patience:]
        best_loss = min(r.loss for r in recent)
        if best_loss == recent[-1].loss:
            return False  # Most recent is still best

        # Check if all recent rounds show no improvement over oldest in window
        oldest_loss = recent[0].loss
        return all(r.loss >= oldest_loss for r in recent[1:])

    # ------------------------------------------------------------------
    # Overfitting detection
    # ------------------------------------------------------------------

    def _check_overfitting(
        self,
        result: ValidationResult,
        training_metrics: dict[str, float],
    ) -> bool:
        """Flag overfitting if training accuracy exceeds validation accuracy by threshold.

        Args:
            result: The current validation result.
            training_metrics: Training phase metrics (expects 'train_accuracy' key).

        Returns:
            True if overfitting is suspected.
        """
        train_accuracy = training_metrics.get("train_accuracy")
        val_accuracy = result.accuracy

        if train_accuracy is None or val_accuracy is None:
            return False

        gap = train_accuracy - val_accuracy
        return gap > self._overfitting_threshold

    # ------------------------------------------------------------------
    # Metric tracking
    # ------------------------------------------------------------------

    async def get_round_metrics(
        self, job_id: str, round_number: int
    ) -> ValidationResult | None:
        """Retrieve validation metrics for a specific round.

        Args:
            job_id: The federated job.
            round_number: The round to look up.

        Returns:
            ValidationResult or None if not found.
        """
        # Check in-memory history first
        for result in self._history.get(job_id, []):
            if result.round_number == round_number:
                return result

        return await self._result_repo.get_validation_result(
            job_id=job_id, round_number=round_number
        )

    async def get_metric_history(
        self,
        job_id: str,
        metric_name: str = "loss",
    ) -> list[dict[str, Any]]:
        """Return the time-series of a named metric across all rounds.

        Args:
            job_id: The federated job.
            metric_name: The metric to extract ('loss', 'accuracy', etc.).

        Returns:
            List of {round_number, value} dicts ordered by round ascending.
        """
        all_results = await self._result_repo.list_validation_results(job_id=job_id)

        time_series: list[dict[str, Any]] = []
        for result in sorted(all_results, key=lambda r: r.round_number):
            value: float | None = None
            if metric_name == "loss":
                value = result.loss
            elif metric_name == "accuracy":
                value = result.accuracy
            elif metric_name == "f1_score":
                value = result.f1_score
            elif metric_name == "auc_roc":
                value = result.auc_roc
            else:
                value = result.custom_metrics.get(metric_name)

            time_series.append(
                {"round_number": result.round_number, "value": value}
            )

        return time_series

    # ------------------------------------------------------------------
    # Baseline comparison
    # ------------------------------------------------------------------

    async def compare_with_baseline(
        self, job_id: str
    ) -> dict[str, Any]:
        """Compare the latest round metrics against the pre-training baseline.

        Args:
            job_id: The federated job to compare.

        Returns:
            Dict with 'baseline', 'latest', and 'improvements' sub-dicts.
            improvements[metric] = latest - baseline (positive = improvement).

        Raises:
            ValueError: If no validation results exist yet or no baseline set.
        """
        if not self._baseline_metrics:
            raise ValueError("No baseline metrics configured for comparison")

        history = self._history.get(job_id, [])
        if not history:
            all_results = await self._result_repo.list_validation_results(job_id=job_id)
            if not all_results:
                raise ValueError(f"No validation results found for job {job_id}")
            latest = max(all_results, key=lambda r: r.round_number)
        else:
            latest = max(history, key=lambda r: r.round_number)

        latest_metrics = {
            "loss": latest.loss,
            "accuracy": latest.accuracy,
            "f1_score": latest.f1_score,
            "auc_roc": latest.auc_roc,
            **latest.custom_metrics,
        }

        improvements: dict[str, float | None] = {}
        for metric, baseline_val in self._baseline_metrics.items():
            current_val = latest_metrics.get(metric)
            if current_val is not None:
                # For loss: lower is better (negative improvement means better)
                # For accuracy/f1/auc: higher is better
                if metric == "loss":
                    improvements[metric] = baseline_val - current_val
                else:
                    improvements[metric] = current_val - baseline_val
            else:
                improvements[metric] = None

        return {
            "baseline": self._baseline_metrics,
            "latest": {k: v for k, v in latest_metrics.items() if v is not None},
            "improvements": improvements,
            "latest_round": latest.round_number,
        }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    async def generate_validation_report(self, job_id: str) -> dict[str, Any]:
        """Generate a structured report of all validation results for a job.

        Args:
            job_id: The federated job.

        Returns:
            Structured report dict with summary statistics and per-round data.
        """
        all_results = await self._result_repo.list_validation_results(job_id=job_id)
        sorted_results = sorted(all_results, key=lambda r: r.round_number)

        if not sorted_results:
            return {
                "job_id": job_id,
                "total_rounds_evaluated": 0,
                "rounds": [],
            }

        losses = [r.loss for r in sorted_results]
        accuracies = [r.accuracy for r in sorted_results if r.accuracy is not None]

        best_loss_result = min(sorted_results, key=lambda r: r.loss)
        convergence_rounds = [r.round_number for r in sorted_results if r.converged]
        overfitting_rounds = [
            r.round_number for r in sorted_results if r.overfitting_detected
        ]

        return {
            "job_id": job_id,
            "total_rounds_evaluated": len(sorted_results),
            "summary": {
                "best_loss": round(min(losses), 6),
                "best_loss_round": best_loss_result.round_number,
                "final_loss": round(losses[-1], 6),
                "loss_improvement": round(losses[0] - losses[-1], 6) if len(losses) > 1 else 0.0,
                "best_accuracy": round(max(accuracies), 4) if accuracies else None,
                "final_accuracy": round(accuracies[-1], 4) if accuracies else None,
                "convergence_detected_at_rounds": convergence_rounds,
                "overfitting_detected_at_rounds": overfitting_rounds,
            },
            "rounds": [r.to_dict() for r in sorted_results],
        }
