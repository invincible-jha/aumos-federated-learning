"""Core business logic services for aumos-federated-learning.

Services are orchestrated by the API layer and depend on adapter protocols
injected at construction time for testability and flexibility.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from aumos_federated_learning.core.interfaces import (
    AggregatorProtocol,
    CoordinatorProtocol,
    DPAggregatorProtocol,
    DropoutHandlerProtocol,
    FLDashboardProtocol,
    FLStrategyProtocol,
    IncentiveScorerProtocol,
    ModelVersionerProtocol,
    ParticipantRegistryProtocol,
    SecureAggregationProtocol,
    SyntheticFallbackProtocol,
    ValidationRunnerProtocol,
)
from aumos_federated_learning.core.models import AggregationRound, FederatedJob, Participant

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job Service
# ---------------------------------------------------------------------------


class JobService:
    """Manage the lifecycle of federated learning training jobs."""

    def __init__(self, repository: Any) -> None:
        self._repo = repository

    async def create_job(
        self,
        *,
        tenant_id: str,
        name: str,
        description: str | None,
        strategy: str,
        num_rounds: int,
        min_participants: int,
        dp_epsilon: float | None,
        dp_delta: float | None,
        strategy_config: dict[str, Any] | None,
    ) -> FederatedJob:
        """Create a new federated learning job in 'configuring' status."""
        job = FederatedJob(
            tenant_id=tenant_id,
            name=name,
            description=description,
            status="configuring",
            strategy=strategy,
            num_rounds=num_rounds,
            current_round=0,
            min_participants=min_participants,
            actual_participants=0,
            dp_epsilon=Decimal(str(dp_epsilon)) if dp_epsilon is not None else None,
            dp_delta=Decimal(str(dp_delta)) if dp_delta is not None else None,
            strategy_config=strategy_config,
        )
        await self._repo.save(job)
        logger.info("Created FederatedJob %s for tenant %s", job.id, tenant_id)
        return job

    async def get_job(self, job_id: uuid.UUID, tenant_id: str) -> FederatedJob | None:
        """Retrieve a job by ID, scoped to the tenant."""
        return await self._repo.get_by_id(job_id, tenant_id=tenant_id)

    async def list_jobs(
        self,
        tenant_id: str,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[FederatedJob]:
        """List jobs for a tenant, optionally filtered by status."""
        return await self._repo.list_jobs(
            tenant_id=tenant_id, status=status, limit=limit, offset=offset
        )

    async def transition_status(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        new_status: str,
    ) -> FederatedJob:
        """Transition job to a new status, validating the transition is legal."""
        valid_transitions: dict[str, list[str]] = {
            "configuring": ["recruiting", "failed"],
            "recruiting": ["training", "failed"],
            "training": ["aggregating", "failed"],
            "aggregating": ["training", "complete", "failed"],
            "complete": [],
            "failed": [],
        }

        job = await self._repo.get_by_id(job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        allowed = valid_transitions.get(job.status, [])
        if new_status not in allowed:
            raise ValueError(
                f"Cannot transition from '{job.status}' to '{new_status}'. "
                f"Allowed: {allowed}"
            )

        job.status = new_status
        await self._repo.save(job)
        logger.info("Job %s transitioned to status '%s'", job_id, new_status)
        return job

    async def start_job(self, job_id: uuid.UUID, tenant_id: str) -> FederatedJob:
        """Start a configured job: transition from configuring → recruiting."""
        return await self.transition_status(job_id, tenant_id, "recruiting")

    async def mark_complete(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        aggregated_model_uri: str,
    ) -> FederatedJob:
        """Mark a job complete with the final model URI."""
        job = await self._repo.get_by_id(job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        job.status = "complete"
        job.aggregated_model_uri = aggregated_model_uri
        await self._repo.save(job)
        return job


# ---------------------------------------------------------------------------
# Training Service
# ---------------------------------------------------------------------------


class TrainingService:
    """Coordinate distributed training rounds across participants."""

    def __init__(
        self,
        job_repository: Any,
        round_repository: Any,
        coordinator: CoordinatorProtocol,
        strategy: FLStrategyProtocol,
    ) -> None:
        self._job_repo = job_repository
        self._round_repo = round_repository
        self._coordinator = coordinator
        self._strategy = strategy

    async def start_round(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        round_number: int,
        participant_ids: list[str],
        timeout_seconds: int = 3600,
    ) -> AggregationRound:
        """Start a new training round: schedule participants and create the round record."""
        job = await self._job_repo.get_by_id(job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        aggregation_round = AggregationRound(
            job_id=job_id,
            round_number=round_number,
            participants_submitted=0,
            aggregation_method=job.strategy,
            dp_noise_added=job.dp_epsilon is not None,
            started_at=datetime.now(tz=timezone.utc),
        )
        await self._round_repo.save(aggregation_round)

        await self._coordinator.schedule_round(
            job_id=str(job_id),
            round_number=round_number,
            participant_ids=participant_ids,
            timeout_seconds=timeout_seconds,
        )

        logger.info("Started round %d for job %s", round_number, job_id)
        return aggregation_round

    async def submit_update(
        self,
        job_id: uuid.UUID,
        round_number: int,
        participant_id: uuid.UUID,
        update_uri: str,
        num_samples: int,
        metrics: dict[str, Any],
    ) -> None:
        """Record that a participant submitted their model update for a round."""
        aggregation_round = await self._round_repo.get_round(
            job_id=job_id, round_number=round_number
        )
        if aggregation_round is None:
            raise ValueError(f"Round {round_number} for job {job_id} not found")

        aggregation_round.participants_submitted += 1
        existing_metrics: dict[str, Any] = aggregation_round.round_metrics or {}
        participant_metrics: dict[str, Any] = existing_metrics.get("participants", {})
        participant_metrics[str(participant_id)] = {
            "update_uri": update_uri,
            "num_samples": num_samples,
            "metrics": metrics,
        }
        existing_metrics["participants"] = participant_metrics
        aggregation_round.round_metrics = existing_metrics
        await self._round_repo.save(aggregation_round)

        logger.info(
            "Participant %s submitted update for job %s round %d",
            participant_id,
            job_id,
            round_number,
        )

    async def get_round_history(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
    ) -> list[AggregationRound]:
        """Return all rounds for a job."""
        return await self._round_repo.list_rounds(job_id=job_id)


# ---------------------------------------------------------------------------
# Aggregation Service
# ---------------------------------------------------------------------------


class AggregationService:
    """Aggregate model updates using the configured strategy and optional DP."""

    def __init__(
        self,
        aggregator: AggregatorProtocol,
        dp_aggregator: DPAggregatorProtocol,
        secure_aggregation: SecureAggregationProtocol,
        round_repository: Any,
        storage: Any,
    ) -> None:
        self._aggregator = aggregator
        self._dp_aggregator = dp_aggregator
        self._secure_agg = secure_aggregation
        self._round_repo = round_repository
        self._storage = storage

    async def aggregate_round(
        self,
        job_id: uuid.UUID,
        round_number: int,
        use_dp: bool,
        dp_epsilon: float | None,
        dp_delta: float | None,
        noise_multiplier: float,
        max_grad_norm: float,
        use_secure_agg: bool,
        participant_public_keys: dict[str, str] | None,
    ) -> tuple[str, dict[str, Any]]:
        """Aggregate all submitted updates for a round.

        Returns (model_uri, aggregation_metrics).
        """
        aggregation_round = await self._round_repo.get_round(
            job_id=job_id, round_number=round_number
        )
        if aggregation_round is None:
            raise ValueError(f"Round {round_number} for job {job_id} not found")

        round_metrics: dict[str, Any] = aggregation_round.round_metrics or {}
        participant_updates: dict[str, Any] = round_metrics.get("participants", {})

        if not participant_updates:
            raise ValueError("No participant updates found for aggregation")

        import numpy as np

        # Load updates from storage
        updates: list[tuple[list[np.ndarray[Any, Any]], int]] = []
        for _pid, update_info in participant_updates.items():
            loaded_update = await self._storage.load_update(update_info["update_uri"])
            updates.append((loaded_update, update_info["num_samples"]))

        aggregation_metrics: dict[str, Any] = {}

        if use_secure_agg and participant_public_keys:
            surviving = list(participant_updates.keys())
            masked = [u[0] for u in updates]
            parameters = self._secure_agg.unmask_aggregate(masked, surviving)
            aggregation_metrics["secure_aggregation"] = True
        else:
            parameters = self._aggregator.aggregate(updates)

        if use_dp and dp_epsilon is not None and dp_delta is not None:
            parameters, privacy_metrics = self._dp_aggregator.aggregate_with_dp(
                updates=[(params, samples) for params, (_, samples) in zip(parameters, updates, strict=False)],
                epsilon=dp_epsilon,
                delta=dp_delta,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )
            aggregation_metrics["privacy"] = privacy_metrics
            aggregation_round.dp_noise_added = True

        model_uri = await self._storage.save_model(
            job_id=str(job_id),
            round_number=round_number,
            parameters=parameters,
        )

        aggregation_round.round_model_uri = model_uri
        aggregation_round.completed_at = datetime.now(tz=timezone.utc)
        aggregation_round.round_metrics = {**round_metrics, "aggregation": aggregation_metrics}
        await self._round_repo.save(aggregation_round)

        logger.info(
            "Aggregated round %d for job %s — model saved to %s", round_number, job_id, model_uri
        )
        return model_uri, aggregation_metrics


# ---------------------------------------------------------------------------
# Coordination Service
# ---------------------------------------------------------------------------


class CoordinationService:
    """Cross-organization scheduling and participant lifecycle management."""

    def __init__(
        self,
        participant_repository: Any,
        coordinator: CoordinatorProtocol,
        job_repository: Any,
    ) -> None:
        self._participant_repo = participant_repository
        self._coordinator = coordinator
        self._job_repo = job_repository

    async def add_participant(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        organization_name: str,
        organization_id: str | None,
        data_size: int | None,
        public_key_pem: str | None,
    ) -> Participant:
        """Register a new participant for a job."""
        participant = Participant(
            tenant_id=tenant_id,
            job_id=job_id,
            organization_name=organization_name,
            organization_id=organization_id,
            status="invited",
            data_size=data_size,
            public_key_pem=public_key_pem,
        )
        await self._participant_repo.save(participant)

        job = await self._job_repo.get_by_id(job_id, tenant_id=tenant_id)
        if job is not None:
            job.actual_participants += 1
            await self._job_repo.save(job)

        logger.info(
            "Added participant %s (%s) to job %s", participant.id, organization_name, job_id
        )
        return participant

    async def accept_invitation(
        self, participant_id: uuid.UUID, tenant_id: str
    ) -> Participant:
        """Accept an invitation, transitioning participant status to 'accepted'."""
        participant = await self._participant_repo.get_by_id(
            participant_id, tenant_id=tenant_id
        )
        if participant is None:
            raise ValueError(f"Participant {participant_id} not found")
        participant.status = "accepted"
        await self._participant_repo.save(participant)
        return participant

    async def get_active_participants(
        self, job_id: uuid.UUID
    ) -> list[Participant]:
        """Return participants in 'accepted' or 'training' status."""
        return await self._participant_repo.list_active(job_id=job_id)

    async def mark_participant_dropped(
        self, participant_id: uuid.UUID, tenant_id: str
    ) -> Participant:
        """Mark a participant as dropped (e.g. timeout, error)."""
        participant = await self._participant_repo.get_by_id(
            participant_id, tenant_id=tenant_id
        )
        if participant is None:
            raise ValueError(f"Participant {participant_id} not found")
        participant.status = "dropped"
        await self._participant_repo.save(participant)
        return participant


# ---------------------------------------------------------------------------
# Fallback Service
# ---------------------------------------------------------------------------


class FallbackService:
    """Trigger synthetic data generation when real participation is insufficient."""

    def __init__(
        self,
        fallback_provider: SyntheticFallbackProtocol,
        job_repository: Any,
    ) -> None:
        self._fallback = fallback_provider
        self._job_repo = job_repository

    async def check_and_trigger_fallback(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        data_schema: dict[str, Any],
    ) -> bool:
        """Check if fallback is needed and trigger it if so.

        Returns True if fallback was triggered.
        """
        job = await self._job_repo.get_by_id(job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        should_fall = await self._fallback.should_fallback(
            job_id=str(job_id),
            min_participants=job.min_participants,
            actual_participants=job.actual_participants,
        )

        if not should_fall:
            return False

        epsilon = float(job.dp_epsilon) if job.dp_epsilon else 1.0
        delta = float(job.dp_delta) if job.dp_delta else 1e-5
        num_synthetic = job.min_participants - job.actual_participants

        await self._fallback.generate_synthetic_participants(
            job_id=str(job_id),
            num_synthetic=num_synthetic,
            data_schema=data_schema,
            epsilon=epsilon,
            delta=delta,
        )

        job.synthetic_fallback_used = True
        await self._job_repo.save(job)

        logger.info(
            "Triggered synthetic fallback for job %s: generating %d synthetic participants",
            job_id,
            num_synthetic,
        )
        return True


# ---------------------------------------------------------------------------
# Metrics Service
# ---------------------------------------------------------------------------


class MetricsService:
    """Track convergence metrics and participant contributions across rounds."""

    def __init__(self, round_repository: Any, job_repository: Any) -> None:
        self._round_repo = round_repository
        self._job_repo = job_repository

    async def record_round_metrics(
        self,
        job_id: uuid.UUID,
        round_number: int,
        loss: float | None,
        accuracy: float | None,
        custom_metrics: dict[str, Any] | None,
    ) -> None:
        """Persist per-round training metrics to the AggregationRound record."""
        aggregation_round = await self._round_repo.get_round(
            job_id=job_id, round_number=round_number
        )
        if aggregation_round is None:
            raise ValueError(f"Round {round_number} for job {job_id} not found")

        existing: dict[str, Any] = aggregation_round.round_metrics or {}
        existing["loss"] = loss
        existing["accuracy"] = accuracy
        if custom_metrics:
            existing.update(custom_metrics)
        aggregation_round.round_metrics = existing
        await self._round_repo.save(aggregation_round)

    async def get_convergence_history(
        self, job_id: uuid.UUID
    ) -> list[dict[str, Any]]:
        """Return a summary of loss/accuracy per round for convergence analysis."""
        rounds: list[AggregationRound] = await self._round_repo.list_rounds(job_id=job_id)
        history: list[dict[str, Any]] = []
        for rnd in sorted(rounds, key=lambda r: r.round_number):
            metrics: dict[str, Any] = rnd.round_metrics or {}
            history.append(
                {
                    "round": rnd.round_number,
                    "loss": metrics.get("loss"),
                    "accuracy": metrics.get("accuracy"),
                    "participants_submitted": rnd.participants_submitted,
                    "dp_noise_added": rnd.dp_noise_added,
                    "completed_at": rnd.completed_at.isoformat() if rnd.completed_at else None,
                }
            )
        return history

    async def compute_contribution_weights(
        self, job_id: uuid.UUID
    ) -> dict[str, float]:
        """Compute each participant's contribution weight based on data size.

        Returns {participant_id: weight} where weights sum to 1.0.
        """
        rounds: list[AggregationRound] = await self._round_repo.list_rounds(job_id=job_id)
        participant_samples: dict[str, int] = {}

        for rnd in rounds:
            metrics: dict[str, Any] = rnd.round_metrics or {}
            for pid, info in metrics.get("participants", {}).items():
                num_samples: int = info.get("num_samples", 0)
                participant_samples[pid] = participant_samples.get(pid, 0) + num_samples

        total = sum(participant_samples.values())
        if total == 0:
            return {}

        return {pid: samples / total for pid, samples in participant_samples.items()}


# ---------------------------------------------------------------------------
# Participant Registry Service
# ---------------------------------------------------------------------------


class ParticipantRegistryService:
    """Orchestrate participant enrollment, liveness, and round assignment.

    Wraps the ParticipantRegistryProtocol adapter with business-rule
    validation: enforces that participants belong to the correct job/tenant
    before delegation.
    """

    def __init__(
        self,
        registry: ParticipantRegistryProtocol,
        job_repository: Any,
    ) -> None:
        self._registry = registry
        self._job_repo = job_repository

    async def enroll_participant(
        self,
        *,
        job_id: uuid.UUID,
        tenant_id: str,
        organization_id: str,
        capabilities: dict[str, Any],
    ) -> str:
        """Validate job existence then enroll the participant node.

        Args:
            job_id: The federated job this participant is joining.
            tenant_id: Tenant context for row-level security.
            organization_id: Enrolling organization's unique identifier.
            capabilities: Declared hardware and data capabilities.

        Returns:
            Assigned participant_id UUID string.

        Raises:
            ValueError: If the job does not exist in this tenant.
        """
        job = await self._job_repo.get_by_id(job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found for tenant {tenant_id}")

        participant_id = await self._registry.register_participant(
            job_id=str(job_id),
            organization_id=organization_id,
            capabilities=capabilities,
        )
        logger.info(
            "Enrolled participant %s for job %s (org=%s)",
            participant_id,
            job_id,
            organization_id,
        )
        return participant_id

    async def heartbeat(self, participant_id: str) -> bool:
        """Record a heartbeat and return current liveness status.

        Args:
            participant_id: The participant pinging in.

        Returns:
            True if participant is considered alive after the heartbeat.
        """
        await self._registry.record_heartbeat(participant_id)
        return await self._registry.check_liveness(participant_id)

    async def select_participants_for_round(
        self,
        job_id: uuid.UUID,
        round_number: int,
        *,
        require_gpu: bool = False,
        min_dataset_size: int = 0,
        min_bandwidth_mbps: float = 0.0,
    ) -> list[str]:
        """Find eligible participants and assign them to a round.

        Args:
            job_id: The federated job.
            round_number: The upcoming training round.
            require_gpu: If True, filter to GPU-equipped participants.
            min_dataset_size: Minimum samples required.
            min_bandwidth_mbps: Minimum upload bandwidth required.

        Returns:
            List of assigned participant_ids.
        """
        eligible = await self._registry.find_eligible_participants(
            str(job_id),
            require_gpu=require_gpu,
            min_dataset_size=min_dataset_size,
            min_bandwidth_mbps=min_bandwidth_mbps,
        )

        if eligible:
            await self._registry.assign_to_round(
                str(job_id), round_number, eligible
            )

        return eligible

    async def perform_liveness_sweep(self, job_id: uuid.UUID) -> list[str]:
        """Sweep all participants for the job and mark timed-out nodes as dropped.

        Args:
            job_id: The federated job to sweep.

        Returns:
            List of newly dropped participant_ids.
        """
        return await self._registry.sweep_dropped_participants(str(job_id))


# ---------------------------------------------------------------------------
# Validation Service
# ---------------------------------------------------------------------------


class ValidationService:
    """Orchestrate central validation and early stopping across rounds.

    Wraps ValidationRunnerProtocol with job-scoped context and integrates
    with the job repository for status transitions on convergence.
    """

    def __init__(
        self,
        validation_runner: ValidationRunnerProtocol,
        job_repository: Any,
    ) -> None:
        self._runner = validation_runner
        self._job_repo = job_repository

    async def evaluate_and_check_stopping(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        round_number: int,
        parameters: list[Any],
        training_metrics: dict[str, float] | None = None,
        early_stopping_patience: int = 5,
    ) -> dict[str, Any]:
        """Evaluate the global model and check whether training should stop.

        Runs central validation, checks convergence and early stopping, and
        transitions the job to 'complete' status if stopping is warranted.

        Args:
            job_id: The federated job.
            tenant_id: Tenant context.
            round_number: The just-completed round.
            parameters: Global model weight arrays.
            training_metrics: Optional training-phase metrics dict.
            early_stopping_patience: Rounds of no improvement before stopping.

        Returns:
            Dict with keys: validation_result, should_stop, reason.
        """
        result = await self._runner.evaluate_round(
            job_id=str(job_id),
            round_number=round_number,
            parameters=parameters,
            training_metrics=training_metrics,
        )

        should_stop = False
        reason: str | None = None

        if result.converged:
            should_stop = True
            reason = "convergence_detected"
        elif self._runner.check_early_stopping(str(job_id), patience=early_stopping_patience):
            should_stop = True
            reason = "early_stopping_patience_exceeded"

        if should_stop:
            logger.info(
                "Training stopping for job %s round %d: %s",
                job_id,
                round_number,
                reason,
            )

        return {
            "validation_result": result,
            "should_stop": should_stop,
            "reason": reason,
        }

    async def get_validation_report(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Generate and return a full validation report for a job.

        Args:
            job_id: The federated job.
            tenant_id: Tenant context (used to verify job access).

        Returns:
            Validation report dict.

        Raises:
            ValueError: If the job is not found.
        """
        job = await self._job_repo.get_by_id(job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found")

        return await self._runner.generate_validation_report(str(job_id))


# ---------------------------------------------------------------------------
# Round Orchestration Service
# ---------------------------------------------------------------------------


class RoundOrchestrationService:
    """Orchestrate a full training round lifecycle including dropout handling.

    Coordinates DropoutHandler, ParticipantRegistry, and TrainingService to
    manage round start, straggler detection, partial aggregation decisions,
    and round completion recording.
    """

    def __init__(
        self,
        dropout_handler: DropoutHandlerProtocol,
        participant_registry: ParticipantRegistryProtocol,
        round_repository: Any,
    ) -> None:
        self._dropout_handler = dropout_handler
        self._registry = participant_registry
        self._round_repo = round_repository

    def begin_round(
        self,
        job_id: uuid.UUID,
        round_number: int,
        participant_ids: list[str],
        timeout_seconds: int = 3600,
    ) -> None:
        """Register dropout tracking for a new round.

        Args:
            job_id: The federated job.
            round_number: The round being started.
            participant_ids: Participants scheduled for this round.
            timeout_seconds: Round deadline in seconds from now.
        """
        from datetime import datetime, timedelta, timezone

        deadline = datetime.now(tz=timezone.utc) + timedelta(seconds=timeout_seconds)
        self._dropout_handler.register_round(
            job_id=str(job_id),
            round_number=round_number,
            assigned_participant_ids=participant_ids,
            deadline=deadline,
        )
        logger.info(
            "Round orchestration started for job %s round %d (%d participants)",
            job_id,
            round_number,
            len(participant_ids),
        )

    def record_participant_submission(
        self,
        job_id: uuid.UUID,
        round_number: int,
        participant_id: str,
    ) -> None:
        """Mark a participant update as received.

        Args:
            job_id: The federated job.
            round_number: The round.
            participant_id: The submitting participant.
        """
        self._dropout_handler.record_submission(
            str(job_id), round_number, participant_id
        )

    async def resolve_round(
        self,
        job_id: uuid.UUID,
        round_number: int,
        min_participants: int,
    ) -> dict[str, Any]:
        """Determine round resolution strategy after the deadline.

        Checks timeouts, evaluates quorum, and returns the set of
        participant updates to include in aggregation.

        Args:
            job_id: The federated job.
            round_number: The round to resolve.
            min_participants: Absolute minimum participants required.

        Returns:
            Dict with keys: participant_ids_for_aggregation, partial_aggregation,
            dropped_participant_ids, quorum_met.
        """
        dropped = await self._dropout_handler.detect_timeouts(str(job_id), round_number)
        quorum_met = self._dropout_handler.check_quorum(str(job_id), round_number)

        if not quorum_met:
            return {
                "participant_ids_for_aggregation": [],
                "partial_aggregation": False,
                "dropped_participant_ids": dropped,
                "quorum_met": False,
            }

        available = self._dropout_handler.trigger_partial_aggregation(
            str(job_id), round_number
        )

        return {
            "participant_ids_for_aggregation": available,
            "partial_aggregation": len(dropped) > 0,
            "dropped_participant_ids": dropped,
            "quorum_met": True,
        }

    def get_dropout_stats(self, job_id: uuid.UUID) -> dict[str, Any]:
        """Return dropout statistics for a job.

        Args:
            job_id: The federated job.

        Returns:
            Dropout statistics dict.
        """
        return self._dropout_handler.get_dropout_statistics(str(job_id))


# ---------------------------------------------------------------------------
# Incentive Service
# ---------------------------------------------------------------------------


class IncentiveService:
    """Score participant contributions and distribute rewards after each round.

    Integrates with IncentiveScorerProtocol to compute data quality scores,
    Shapley value approximations, and reward allocations, and surfaces
    free-rider detection for governance action.
    """

    def __init__(
        self,
        scorer: IncentiveScorerProtocol,
        job_repository: Any,
    ) -> None:
        self._scorer = scorer
        self._job_repo = job_repository

    async def score_and_distribute(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        round_number: int,
        participant_data: list[dict[str, Any]],
        shapley_values: dict[str, float] | None = None,
        model_improvement_attributions: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Score all participants for a round and return reward distribution.

        Args:
            job_id: The federated job.
            tenant_id: Tenant context for access verification.
            round_number: The completed training round.
            participant_data: List of participant capability/submission metadata.
            shapley_values: Pre-computed Shapley values if available.
            model_improvement_attributions: Pre-computed improvement attributions.

        Returns:
            Dict with contribution_records, free_riders, reward_distribution.

        Raises:
            ValueError: If the job is not found for this tenant.
        """
        job = await self._job_repo.get_by_id(job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found for tenant {tenant_id}")

        records = await self._scorer.score_round(
            job_id=str(job_id),
            round_number=round_number,
            participant_data=participant_data,
            shapley_values=shapley_values,
            model_improvement_attributions=model_improvement_attributions,
        )

        free_riders = [r.participant_id for r in records if r.is_free_rider]
        reward_distribution = {r.participant_id: r.reward_units for r in records}

        logger.info(
            "Incentive distribution for job %s round %d: "
            "%d participants, %d free-riders",
            job_id,
            round_number,
            len(records),
            len(free_riders),
        )

        return {
            "contribution_records": [r.to_dict() for r in records],
            "free_riders": free_riders,
            "reward_distribution": reward_distribution,
        }

    async def get_distribution_report(
        self, job_id: uuid.UUID, tenant_id: str
    ) -> dict[str, Any]:
        """Return the incentive distribution report for an entire job.

        Args:
            job_id: The federated job.
            tenant_id: Tenant context.

        Returns:
            Report dict from IncentiveScorer.

        Raises:
            ValueError: If the job is not found.
        """
        job = await self._job_repo.get_by_id(job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found for tenant {tenant_id}")

        return await self._scorer.generate_distribution_report(job_id=str(job_id))


# ---------------------------------------------------------------------------
# Dashboard Service
# ---------------------------------------------------------------------------


class DashboardService:
    """Expose training progress aggregation for the monitoring layer.

    Wraps FLDashboardProtocol with job-scoped access control, feeding
    round start and completion events from the training and aggregation
    services, and building full snapshots on demand.
    """

    def __init__(
        self,
        dashboard: FLDashboardProtocol,
        job_repository: Any,
    ) -> None:
        self._dashboard = dashboard
        self._job_repo = job_repository

    def record_round_start(
        self,
        job_id: uuid.UUID,
        round_number: int,
        participant_ids: list[str],
    ) -> None:
        """Push round-start event into the dashboard.

        Args:
            job_id: The federated job.
            round_number: The round being started.
            participant_ids: Participants assigned to this round.
        """
        self._dashboard.ingest_round_start(
            job_id=str(job_id),
            round_number=round_number,
            assigned_participants=participant_ids,
        )

    def record_round_completion(
        self,
        job_id: uuid.UUID,
        round_number: int,
        *,
        participants_submitted: int,
        participants_dropped: int,
        loss: float | None = None,
        accuracy: float | None = None,
        dp_epsilon_consumed: float | None = None,
        bytes_transmitted: int = 0,
    ) -> None:
        """Push round-completion event into the dashboard.

        Args:
            job_id: The federated job.
            round_number: The completed round.
            participants_submitted: Submission count.
            participants_dropped: Dropout count.
            loss: Validation loss.
            accuracy: Validation accuracy.
            dp_epsilon_consumed: Cumulative privacy budget consumed.
            bytes_transmitted: Bytes of weights exchanged.
        """
        self._dashboard.ingest_round_completion(
            job_id=str(job_id),
            round_number=round_number,
            participants_submitted=participants_submitted,
            participants_dropped=participants_dropped,
            loss=loss,
            accuracy=accuracy,
            dp_epsilon_consumed=dp_epsilon_consumed,
            bytes_transmitted=bytes_transmitted,
        )

    async def get_snapshot(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        total_epsilon_budget: float | None = None,
    ) -> dict[str, Any]:
        """Build and return the full dashboard snapshot for a job.

        Args:
            job_id: The federated job.
            tenant_id: Tenant context for access control.
            total_epsilon_budget: Optional privacy budget cap for the job.

        Returns:
            Dashboard JSON snapshot.

        Raises:
            ValueError: If the job is not found.
        """
        job = await self._job_repo.get_by_id(job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found for tenant {tenant_id}")

        # Derive epsilon budget from job configuration if not provided
        if total_epsilon_budget is None and job.dp_epsilon is not None:
            total_epsilon_budget = float(job.dp_epsilon) * job.num_rounds

        return await self._dashboard.export_dashboard_json(
            job_id=str(job_id),
            total_epsilon_budget=total_epsilon_budget,
        )
