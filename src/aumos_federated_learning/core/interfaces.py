"""Protocol interfaces for the federated learning hexagonal architecture.

These protocols define the contracts that adapters must implement, enabling
dependency injection and easy testing/swapping of implementations.
"""

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
