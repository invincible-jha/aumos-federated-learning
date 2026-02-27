"""Protocol interfaces for the federated learning hexagonal architecture.

These protocols define the contracts that adapters must implement, enabling
dependency injection and easy testing/swapping of implementations.
"""

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FLStrategyProtocol(Protocol):
    """Protocol for federated learning aggregation strategies (FedAvg, FedProx, SCAFFOLD)."""

    @property
    def strategy_name(self) -> str:
        """Return the canonical name of this strategy."""
        ...

    def configure_fit(
        self,
        server_round: int,
        parameters: list[np.ndarray[Any, Any]],
        client_manager: Any,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Configure the next round of distributed training.

        Returns a list of (client, fit_ins) tuples for sampled clients.
        """
        ...

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[Any, Any]],
        failures: list[Any],
    ) -> tuple[list[np.ndarray[Any, Any]] | None, dict[str, Any]]:
        """Aggregate model updates from completed clients.

        Returns (aggregated_parameters, aggregated_metrics).
        """
        ...

    def configure_evaluate(
        self,
        server_round: int,
        parameters: list[np.ndarray[Any, Any]],
        client_manager: Any,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Configure optional distributed evaluation round."""
        ...

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[Any, Any]],
        failures: list[Any],
    ) -> tuple[float | None, dict[str, Any]]:
        """Aggregate evaluation results from clients."""
        ...


@runtime_checkable
class AggregatorProtocol(Protocol):
    """Protocol for plain (non-DP) model update aggregation."""

    def aggregate(
        self,
        updates: list[tuple[list[np.ndarray[Any, Any]], int]],
    ) -> list[np.ndarray[Any, Any]]:
        """Aggregate a list of (parameter_arrays, num_samples) pairs.

        Returns weighted-average of parameter arrays.
        """
        ...


@runtime_checkable
class DPAggregatorProtocol(Protocol):
    """Protocol for differentially private aggregation of model updates."""

    def aggregate_with_dp(
        self,
        updates: list[tuple[list[np.ndarray[Any, Any]], int]],
        epsilon: float,
        delta: float,
        noise_multiplier: float,
        max_grad_norm: float,
    ) -> tuple[list[np.ndarray[Any, Any]], dict[str, float]]:
        """Aggregate updates and apply calibrated Gaussian DP noise.

        Returns (noised_parameters, privacy_accounting_metrics).
        """
        ...

    def compute_epsilon(
        self,
        num_rounds: int,
        noise_multiplier: float,
        sample_rate: float,
        delta: float,
    ) -> float:
        """Compute the total privacy budget consumed for a given training configuration."""
        ...


@runtime_checkable
class CoordinatorProtocol(Protocol):
    """Protocol for cross-organization training coordination."""

    async def invite_participant(
        self,
        job_id: str,
        organization_id: str,
        organization_name: str,
        metadata: dict[str, Any],
    ) -> str:
        """Send an invitation to an organization and return the participant_id."""
        ...

    async def schedule_round(
        self,
        job_id: str,
        round_number: int,
        participant_ids: list[str],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Schedule a training round and return scheduling metadata."""
        ...

    async def collect_updates(
        self,
        job_id: str,
        round_number: int,
        timeout_seconds: int,
    ) -> list[dict[str, Any]]:
        """Wait for and collect model updates from participants for a round."""
        ...

    async def broadcast_global_model(
        self,
        job_id: str,
        round_number: int,
        model_uri: str,
        participant_ids: list[str],
    ) -> None:
        """Broadcast the aggregated global model to all participants."""
        ...


@runtime_checkable
class SyntheticFallbackProtocol(Protocol):
    """Protocol for falling back to synthetic data when participation is insufficient."""

    async def should_fallback(
        self,
        job_id: str,
        min_participants: int,
        actual_participants: int,
    ) -> bool:
        """Determine whether synthetic fallback should be triggered."""
        ...

    async def generate_synthetic_participants(
        self,
        job_id: str,
        num_synthetic: int,
        data_schema: dict[str, Any],
        epsilon: float,
        delta: float,
    ) -> list[dict[str, Any]]:
        """Generate synthetic participant data via the privacy engine.

        Returns a list of synthetic participant metadata dicts.
        """
        ...


@runtime_checkable
class SecureAggregationProtocol(Protocol):
    """Protocol for cryptographic (MPC-based) secure aggregation."""

    def setup_round(
        self,
        participant_public_keys: dict[str, str],
        threshold: float,
    ) -> dict[str, Any]:
        """Set up a secure aggregation round, returning per-participant masks."""
        ...

    def unmask_aggregate(
        self,
        masked_aggregates: list[np.ndarray[Any, Any]],
        surviving_participants: list[str],
    ) -> list[np.ndarray[Any, Any]]:
        """Unmask the aggregate once threshold participants have submitted."""
        ...


# ---------------------------------------------------------------------------
# New adapter protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ParticipantRegistryProtocol(Protocol):
    """Protocol for federated participant node enrollment and lifecycle management."""

    async def register_participant(
        self,
        *,
        job_id: str,
        organization_id: str,
        capabilities: dict[str, Any],
    ) -> str:
        """Enroll a new participant node; return the assigned participant_id."""
        ...

    async def record_heartbeat(self, participant_id: str) -> None:
        """Record a liveness heartbeat ping from a participant."""
        ...

    async def check_liveness(self, participant_id: str) -> bool:
        """Return True if the participant heartbeated within the timeout window."""
        ...

    async def find_eligible_participants(
        self,
        job_id: str,
        *,
        require_gpu: bool = False,
        min_dataset_size: int = 0,
        min_bandwidth_mbps: float = 0.0,
        required_framework: str | None = None,
    ) -> list[str]:
        """Return participant_ids matching the stated capability requirements."""
        ...

    async def assign_to_round(
        self,
        job_id: str,
        round_number: int,
        participant_ids: list[str],
    ) -> None:
        """Record round assignment for a set of participants."""
        ...

    async def approve_enrollment(
        self,
        participant_id: str,
        *,
        approved_by: str,
        approval_notes: str | None = None,
    ) -> None:
        """Approve a pending enrollment, transitioning status to idle."""
        ...

    async def get_participant_history(
        self, participant_id: str
    ) -> dict[str, Any]:
        """Return contribution history summary for a participant."""
        ...

    async def sweep_dropped_participants(self, job_id: str) -> list[str]:
        """Mark timed-out participants as dropped; return their IDs."""
        ...


@runtime_checkable
class FederatedCommunicationProtocol(Protocol):
    """Protocol for secure inter-node messaging in a federated topology."""

    async def open_channel(
        self,
        participant_id: str,
        channel_config: Any,
    ) -> None:
        """Open an mTLS gRPC channel to a remote participant."""
        ...

    async def close_channel(self, participant_id: str) -> None:
        """Close the channel for a participant."""
        ...

    async def send_model_update(
        self,
        participant_id: str,
        job_id: str,
        round_number: int,
        parameters: list[np.ndarray[Any, Any]],
        *,
        compression: str | None = None,
    ) -> str:
        """Send aggregated model weights; return the message_id for tracking."""
        ...

    def get_open_channels(self) -> list[str]:
        """Return participant_ids with currently open channels."""
        ...

    def pool_summary(self) -> dict[str, Any]:
        """Return aggregate connection pool statistics."""
        ...


@runtime_checkable
class ModelVersionerProtocol(Protocol):
    """Protocol for federated model checkpoint storage and version management."""

    async def save_checkpoint(
        self,
        job_id: str,
        round_number: int,
        parameters: list[np.ndarray[Any, Any]],
        metrics: dict[str, Any] | None = None,
    ) -> Any:
        """Serialize and persist a global model checkpoint; return version metadata."""
        ...

    async def load_checkpoint(
        self,
        job_id: str,
        round_number: int | None = None,
        version_id: str | None = None,
    ) -> tuple[list[np.ndarray[Any, Any]], Any]:
        """Load model weights from a checkpoint; return (parameters, version_metadata)."""
        ...

    async def get_current_round(self, job_id: str) -> int:
        """Return the most recently completed round number for a job."""
        ...

    async def rollback_to_round(
        self,
        job_id: str,
        target_round: int,
    ) -> Any:
        """Restore the global model to a previous round checkpoint."""
        ...

    async def get_version_history(
        self,
        job_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return version metadata for recent checkpoints, most recent first."""
        ...

    async def compute_round_delta(
        self,
        job_id: str,
        from_round: int,
        to_round: int,
    ) -> list[np.ndarray[Any, Any]]:
        """Return the parameter delta between two model versions."""
        ...


@runtime_checkable
class ValidationRunnerProtocol(Protocol):
    """Protocol for central holdout-set evaluation of the global model."""

    async def evaluate_round(
        self,
        job_id: str,
        round_number: int,
        parameters: list[np.ndarray[Any, Any]],
        training_metrics: dict[str, float] | None = None,
    ) -> Any:
        """Evaluate the global model for a round; return a ValidationResult."""
        ...

    async def get_metric_history(
        self,
        job_id: str,
        metric_name: str = "loss",
    ) -> list[dict[str, Any]]:
        """Return the time-series of a named metric across all rounds."""
        ...

    def check_early_stopping(
        self,
        job_id: str,
        patience: int = 5,
    ) -> bool:
        """Return True if training should stop early due to no improvement."""
        ...

    async def generate_validation_report(self, job_id: str) -> dict[str, Any]:
        """Generate a structured report of all validation results for a job."""
        ...


@runtime_checkable
class DropoutHandlerProtocol(Protocol):
    """Protocol for straggler tolerance and churn recovery in FL rounds."""

    def register_round(
        self,
        job_id: str,
        round_number: int,
        assigned_participant_ids: list[str],
        deadline: datetime | None = None,
    ) -> Any:
        """Register participants and deadline for a training round."""
        ...

    def record_submission(
        self, job_id: str, round_number: int, participant_id: str
    ) -> None:
        """Mark a participant as having submitted their update."""
        ...

    async def detect_timeouts(
        self, job_id: str, round_number: int
    ) -> list[str]:
        """Return participant_ids that have missed the round deadline."""
        ...

    def check_quorum(self, job_id: str, round_number: int) -> bool:
        """Return True if the minimum quorum fraction has been met."""
        ...

    def trigger_partial_aggregation(
        self, job_id: str, round_number: int
    ) -> list[str]:
        """Proceed with partial aggregation; return available participant_ids."""
        ...

    async def extend_round_deadline(
        self,
        job_id: str,
        round_number: int,
    ) -> datetime | None:
        """Extend the round deadline for stragglers; return new deadline."""
        ...

    async def resync_dropped_participant(
        self,
        job_id: str,
        participant_id: str,
        next_round_number: int,
    ) -> bool:
        """Re-enroll a dropped participant for a future round."""
        ...

    def get_dropout_statistics(self, job_id: str) -> dict[str, Any]:
        """Return cumulative dropout statistics for a job."""
        ...


@runtime_checkable
class IncentiveScorerProtocol(Protocol):
    """Protocol for contribution scoring and reward distribution."""

    def score_data_quality(
        self,
        participant_id: str,
        num_samples: int,
        declared_class_distribution: dict[str, int] | None = None,
        duplicate_fraction: float = 0.0,
        missing_value_fraction: float = 0.0,
    ) -> float:
        """Return a data quality score (0–1) for a participant's dataset."""
        ...

    def approximate_shapley_values(
        self,
        participant_ids: list[str],
        characteristic_function: Any,
        num_permutations: int | None = None,
    ) -> dict[str, float]:
        """Compute approximate Shapley values for all participants."""
        ...

    def calculate_rewards(
        self,
        shapley_values: dict[str, float],
        reward_pool: float | None = None,
    ) -> dict[str, float]:
        """Allocate reward units proportional to Shapley values."""
        ...

    def detect_free_riders(
        self,
        shapley_values: dict[str, float],
    ) -> list[str]:
        """Return participant_ids suspected of free-riding."""
        ...

    async def score_round(
        self,
        job_id: str,
        round_number: int,
        participant_data: list[dict[str, Any]],
        shapley_values: dict[str, float] | None = None,
        model_improvement_attributions: dict[str, float] | None = None,
    ) -> list[Any]:
        """Score all participants for a round and persist contribution records."""
        ...

    async def generate_distribution_report(
        self, job_id: str
    ) -> dict[str, Any]:
        """Generate an incentive distribution report for an entire job."""
        ...


@runtime_checkable
class FLDashboardProtocol(Protocol):
    """Protocol for federated training progress aggregation and export."""

    def ingest_round_start(
        self,
        job_id: str,
        round_number: int,
        assigned_participants: list[str],
        started_at: datetime | None = None,
    ) -> None:
        """Record the start of a training round."""
        ...

    def ingest_round_completion(
        self,
        job_id: str,
        round_number: int,
        *,
        participants_submitted: int,
        participants_dropped: int,
        loss: float | None = None,
        accuracy: float | None = None,
        dp_epsilon_consumed: float | None = None,
        bytes_transmitted: int = 0,
        completed_at: datetime | None = None,
    ) -> None:
        """Record the completion of a training round with all key metrics."""
        ...

    def get_loss_curve(self, job_id: str) -> list[dict[str, Any]]:
        """Return training loss curve as a list of {round_number, loss} dicts."""
        ...

    def get_accuracy_curve(self, job_id: str) -> list[dict[str, Any]]:
        """Return accuracy curve as a list of {round_number, accuracy} dicts."""
        ...

    def get_privacy_budget_summary(
        self,
        job_id: str,
        total_epsilon_budget: float | None = None,
    ) -> dict[str, Any]:
        """Return privacy budget consumption summary."""
        ...

    async def export_dashboard_json(
        self,
        job_id: str,
        total_epsilon_budget: float | None = None,
    ) -> dict[str, Any]:
        """Export a complete dashboard snapshot as a JSON-ready dict."""
        ...
