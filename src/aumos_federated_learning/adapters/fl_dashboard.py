"""FL Dashboard adapter — federated training progress aggregation.

Aggregates per-round metrics, participant status, training curves, communication
efficiency, privacy budget consumption, and timing statistics into a
dashboard-ready JSON snapshot. Designed to power real-time monitoring UIs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RoundSummary:
    """Aggregated metrics for a single training round.

    Attributes:
        round_number: The training round identifier.
        started_at: UTC timestamp when the round began.
        completed_at: UTC timestamp when aggregation finished (None if in progress).
        participants_assigned: Total participants scheduled for this round.
        participants_submitted: Participants who submitted updates.
        participants_dropped: Participants who timed out or failed.
        loss: Validation loss after this round (None if not yet evaluated).
        accuracy: Validation accuracy after this round.
        dp_epsilon_consumed: Cumulative privacy budget consumed through this round.
        bytes_transmitted: Total bytes of model weights sent/received this round.
        round_duration_seconds: Wall-clock duration of the round (None if in progress).
    """

    round_number: int
    started_at: datetime | None = None
    completed_at: datetime | None = None
    participants_assigned: int = 0
    participants_submitted: int = 0
    participants_dropped: int = 0
    loss: float | None = None
    accuracy: float | None = None
    dp_epsilon_consumed: float | None = None
    bytes_transmitted: int = 0
    round_duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dashboard-ready dict."""
        duration = None
        if self.started_at and self.completed_at:
            duration = (self.completed_at - self.started_at).total_seconds()
        return {
            "round_number": self.round_number,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "participants_assigned": self.participants_assigned,
            "participants_submitted": self.participants_submitted,
            "participants_dropped": self.participants_dropped,
            "submission_rate": (
                round(self.participants_submitted / self.participants_assigned, 4)
                if self.participants_assigned > 0
                else 0.0
            ),
            "loss": self.loss,
            "accuracy": self.accuracy,
            "dp_epsilon_consumed": self.dp_epsilon_consumed,
            "bytes_transmitted": self.bytes_transmitted,
            "round_duration_seconds": (
                round(duration, 2) if duration is not None else self.round_duration_seconds
            ),
        }


# ---------------------------------------------------------------------------
# FLDashboard
# ---------------------------------------------------------------------------


class FLDashboard:
    """Federated learning training progress aggregation for dashboard display.

    Collects data from validation runners, dropout handlers, communication
    adapters, incentive scorers, and DP aggregators to build a unified
    real-time training snapshot. Snapshots are exportable as JSON for
    consumption by frontend dashboards or monitoring pipelines.

    Args:
        validation_runner: CentralValidationRunner providing per-round metrics.
        dropout_handler: DropoutHandler providing participation statistics.
        communication_adapter: FederatedCommunicationAdapter for bandwidth stats.
        incentive_scorer: IncentiveScorer providing contribution data.
        dp_aggregator: DPAggregatorProtocol for privacy budget tracking.
        snapshot_repository: Async repository for persisting dashboard snapshots.
    """

    def __init__(
        self,
        validation_runner: Any | None = None,
        dropout_handler: Any | None = None,
        communication_adapter: Any | None = None,
        incentive_scorer: Any | None = None,
        dp_aggregator: Any | None = None,
        snapshot_repository: Any | None = None,
    ) -> None:
        self._validation_runner = validation_runner
        self._dropout_handler = dropout_handler
        self._communication_adapter = communication_adapter
        self._incentive_scorer = incentive_scorer
        self._dp_aggregator = dp_aggregator
        self._snapshot_repo = snapshot_repository

        # job_id -> list[RoundSummary] (in-memory accumulation)
        self._round_summaries: dict[str, list[RoundSummary]] = {}

        # job_id -> cumulative dp epsilon consumed
        self._cumulative_epsilon: dict[str, float] = {}

        # job_id -> cumulative bytes (sent + received)
        self._cumulative_bytes: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Round ingestion
    # ------------------------------------------------------------------

    def ingest_round_start(
        self,
        job_id: str,
        round_number: int,
        assigned_participants: list[str],
        started_at: datetime | None = None,
    ) -> None:
        """Record the start of a training round.

        Args:
            job_id: The federated job.
            round_number: The round that started.
            assigned_participants: Participant IDs scheduled for this round.
            started_at: Override the start timestamp (defaults to now).
        """
        summaries = self._round_summaries.setdefault(job_id, [])

        # Avoid duplicates
        for existing in summaries:
            if existing.round_number == round_number:
                existing.started_at = started_at or datetime.now(tz=timezone.utc)
                existing.participants_assigned = len(assigned_participants)
                return

        summary = RoundSummary(
            round_number=round_number,
            started_at=started_at or datetime.now(tz=timezone.utc),
            participants_assigned=len(assigned_participants),
        )
        summaries.append(summary)

        logger.debug(
            "Dashboard: ingested round %d start for job %s (%d participants)",
            round_number,
            job_id,
            len(assigned_participants),
        )

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
        """Record the completion of a training round with all key metrics.

        Args:
            job_id: The federated job.
            round_number: The round that completed.
            participants_submitted: How many participants submitted updates.
            participants_dropped: How many participants dropped out.
            loss: Validation loss for this round.
            accuracy: Validation accuracy for this round.
            dp_epsilon_consumed: Privacy budget consumed cumulatively through this round.
            bytes_transmitted: Total bytes of weights exchanged this round.
            completed_at: Override the completion timestamp.
        """
        summary = self._get_or_create_summary(job_id, round_number)
        summary.completed_at = completed_at or datetime.now(tz=timezone.utc)
        summary.participants_submitted = participants_submitted
        summary.participants_dropped = participants_dropped
        summary.loss = loss
        summary.accuracy = accuracy
        summary.dp_epsilon_consumed = dp_epsilon_consumed
        summary.bytes_transmitted = bytes_transmitted

        # Update job-level cumulative stats
        if dp_epsilon_consumed is not None:
            self._cumulative_epsilon[job_id] = dp_epsilon_consumed
        self._cumulative_bytes[job_id] = (
            self._cumulative_bytes.get(job_id, 0) + bytes_transmitted
        )

        logger.info(
            "Dashboard: round %d completed for job %s "
            "(submitted=%d, dropped=%d, loss=%s, accuracy=%s)",
            round_number,
            job_id,
            participants_submitted,
            participants_dropped,
            f"{loss:.4f}" if loss is not None else "N/A",
            f"{accuracy:.4f}" if accuracy is not None else "N/A",
        )

    # ------------------------------------------------------------------
    # Per-round metrics aggregation
    # ------------------------------------------------------------------

    def get_round_summaries(
        self, job_id: str
    ) -> list[dict[str, Any]]:
        """Return all round summaries for a job, sorted by round number.

        Args:
            job_id: The federated job.

        Returns:
            List of serialized RoundSummary dicts ordered by round ascending.
        """
        summaries = self._round_summaries.get(job_id, [])
        return [
            s.to_dict()
            for s in sorted(summaries, key=lambda s: s.round_number)
        ]

    # ------------------------------------------------------------------
    # Participant status summary
    # ------------------------------------------------------------------

    async def get_participant_status_summary(
        self, job_id: str
    ) -> dict[str, Any]:
        """Aggregate participant status counts across all rounds.

        Delegates to the dropout_handler and participant_registry if available.

        Args:
            job_id: The federated job.

        Returns:
            Dict with active_count, idle_count, dropped_count, total_count,
            and per-round dropout stats.
        """
        summary: dict[str, Any] = {"job_id": job_id}

        if self._dropout_handler is not None:
            dropout_stats = self._dropout_handler.get_dropout_statistics(job_id)
            summary["dropout_stats"] = dropout_stats
        else:
            summary["dropout_stats"] = {}

        # Derive participation counts from ingested round data
        summaries = self._round_summaries.get(job_id, [])
        if summaries:
            latest = max(summaries, key=lambda s: s.round_number)
            summary.update(
                {
                    "latest_round": latest.round_number,
                    "latest_submitted": latest.participants_submitted,
                    "latest_dropped": latest.participants_dropped,
                    "latest_assigned": latest.participants_assigned,
                }
            )

        return summary

    # ------------------------------------------------------------------
    # Training curves
    # ------------------------------------------------------------------

    def get_loss_curve(self, job_id: str) -> list[dict[str, Any]]:
        """Return training loss across all completed rounds.

        Args:
            job_id: The federated job.

        Returns:
            List of {round_number, loss} dicts ordered by round ascending.
        """
        return [
            {"round_number": s.round_number, "loss": s.loss}
            for s in sorted(
                self._round_summaries.get(job_id, []),
                key=lambda s: s.round_number,
            )
            if s.loss is not None
        ]

    def get_accuracy_curve(self, job_id: str) -> list[dict[str, Any]]:
        """Return validation accuracy across all completed rounds.

        Args:
            job_id: The federated job.

        Returns:
            List of {round_number, accuracy} dicts ordered by round ascending.
        """
        return [
            {"round_number": s.round_number, "accuracy": s.accuracy}
            for s in sorted(
                self._round_summaries.get(job_id, []),
                key=lambda s: s.round_number,
            )
            if s.accuracy is not None
        ]

    def get_submission_rate_curve(self, job_id: str) -> list[dict[str, Any]]:
        """Return participant submission rate across all rounds.

        Args:
            job_id: The federated job.

        Returns:
            List of {round_number, submission_rate} dicts.
        """
        result: list[dict[str, Any]] = []
        for s in sorted(
            self._round_summaries.get(job_id, []),
            key=lambda s: s.round_number,
        ):
            if s.participants_assigned > 0:
                rate = s.participants_submitted / s.participants_assigned
            else:
                rate = 0.0
            result.append(
                {"round_number": s.round_number, "submission_rate": round(rate, 4)}
            )
        return result

    # ------------------------------------------------------------------
    # Communication efficiency metrics
    # ------------------------------------------------------------------

    def get_communication_efficiency(
        self, job_id: str
    ) -> dict[str, Any]:
        """Return bandwidth and communication efficiency statistics.

        Args:
            job_id: The federated job.

        Returns:
            Dict with total_bytes, bytes_per_round, rounds_tracked, and
            optionally pool summary from the communication adapter.
        """
        summaries = self._round_summaries.get(job_id, [])
        total_bytes = sum(s.bytes_transmitted for s in summaries)
        num_rounds = len(summaries)
        avg_bytes_per_round = total_bytes / num_rounds if num_rounds > 0 else 0.0

        stats: dict[str, Any] = {
            "job_id": job_id,
            "total_bytes_transmitted": total_bytes,
            "average_bytes_per_round": round(avg_bytes_per_round, 2),
            "rounds_tracked": num_rounds,
        }

        if self._communication_adapter is not None:
            try:
                pool_stats = self._communication_adapter.pool_summary()
                stats["connection_pool"] = pool_stats
            except Exception as exc:
                logger.warning("Failed to fetch communication pool stats: %s", exc)

        return stats

    # ------------------------------------------------------------------
    # Privacy budget tracking
    # ------------------------------------------------------------------

    def get_privacy_budget_summary(
        self,
        job_id: str,
        total_epsilon_budget: float | None = None,
    ) -> dict[str, Any]:
        """Summarise differential privacy budget consumption.

        Args:
            job_id: The federated job.
            total_epsilon_budget: The configured total epsilon budget for the job.
                If provided, includes budget remaining and fraction consumed.

        Returns:
            Dict with cumulative_epsilon, dp_enabled, per-round epsilon history,
            and budget_remaining (if total provided).
        """
        summaries = sorted(
            self._round_summaries.get(job_id, []),
            key=lambda s: s.round_number,
        )

        epsilon_history = [
            {"round_number": s.round_number, "cumulative_epsilon": s.dp_epsilon_consumed}
            for s in summaries
            if s.dp_epsilon_consumed is not None
        ]

        cumulative = self._cumulative_epsilon.get(job_id)
        dp_enabled = cumulative is not None or any(
            s.dp_epsilon_consumed is not None for s in summaries
        )

        result: dict[str, Any] = {
            "job_id": job_id,
            "dp_enabled": dp_enabled,
            "cumulative_epsilon": cumulative,
            "epsilon_history": epsilon_history,
        }

        if total_epsilon_budget is not None and cumulative is not None:
            result["total_epsilon_budget"] = total_epsilon_budget
            result["budget_remaining"] = max(0.0, total_epsilon_budget - cumulative)
            result["fraction_consumed"] = round(
                min(1.0, cumulative / total_epsilon_budget), 4
            )

        return result

    # ------------------------------------------------------------------
    # Timing statistics
    # ------------------------------------------------------------------

    def get_timing_statistics(self, job_id: str) -> dict[str, Any]:
        """Compute round timing statistics for performance analysis.

        Args:
            job_id: The federated job.

        Returns:
            Dict with average_round_duration_seconds, min_duration, max_duration,
            total_training_duration_seconds, and per-round timings.
        """
        summaries = sorted(
            self._round_summaries.get(job_id, []),
            key=lambda s: s.round_number,
        )

        durations: list[float] = []
        for s in summaries:
            if s.started_at and s.completed_at:
                duration = (s.completed_at - s.started_at).total_seconds()
                durations.append(duration)

        per_round_timings = [
            {
                "round_number": s.round_number,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                "duration_seconds": (
                    round((s.completed_at - s.started_at).total_seconds(), 2)
                    if s.started_at and s.completed_at
                    else None
                ),
            }
            for s in summaries
        ]

        return {
            "job_id": job_id,
            "rounds_completed": len(durations),
            "average_round_duration_seconds": (
                round(sum(durations) / len(durations), 2) if durations else None
            ),
            "min_round_duration_seconds": round(min(durations), 2) if durations else None,
            "max_round_duration_seconds": round(max(durations), 2) if durations else None,
            "total_training_duration_seconds": (
                round(sum(durations), 2) if durations else None
            ),
            "per_round_timings": per_round_timings,
        }

    # ------------------------------------------------------------------
    # Full dashboard snapshot export
    # ------------------------------------------------------------------

    async def export_dashboard_json(
        self,
        job_id: str,
        total_epsilon_budget: float | None = None,
    ) -> dict[str, Any]:
        """Build and return a complete dashboard-ready JSON snapshot.

        Aggregates all sub-metrics into a single nested dict suitable for
        API responses or direct rendering in a monitoring frontend.

        Args:
            job_id: The federated job.
            total_epsilon_budget: Optional privacy budget cap for the job.

        Returns:
            Comprehensive dashboard snapshot dict.
        """
        now = datetime.now(tz=timezone.utc)
        summaries = self._round_summaries.get(job_id, [])
        num_rounds = len(summaries)

        snapshot: dict[str, Any] = {
            "job_id": job_id,
            "snapshot_at": now.isoformat(),
            "total_rounds_tracked": num_rounds,
            "rounds": self.get_round_summaries(job_id),
            "loss_curve": self.get_loss_curve(job_id),
            "accuracy_curve": self.get_accuracy_curve(job_id),
            "submission_rate_curve": self.get_submission_rate_curve(job_id),
            "participant_status": await self.get_participant_status_summary(job_id),
            "communication_efficiency": self.get_communication_efficiency(job_id),
            "privacy_budget": self.get_privacy_budget_summary(
                job_id, total_epsilon_budget=total_epsilon_budget
            ),
            "timing": self.get_timing_statistics(job_id),
        }

        # Optionally include incentive distribution summary
        if self._incentive_scorer is not None:
            try:
                incentive_report = await self._incentive_scorer.generate_distribution_report(
                    job_id=job_id
                )
                snapshot["incentive_distribution"] = incentive_report
            except Exception as exc:
                logger.warning("Failed to fetch incentive report: %s", exc)
                snapshot["incentive_distribution"] = None

        # Persist snapshot if a repository is configured
        if self._snapshot_repo is not None:
            try:
                await self._snapshot_repo.save_snapshot(
                    job_id=job_id,
                    snapshot_at=now,
                    snapshot_data=snapshot,
                )
            except Exception as exc:
                logger.warning("Failed to persist dashboard snapshot: %s", exc)

        logger.info(
            "Exported dashboard snapshot for job %s (%d rounds)", job_id, num_rounds
        )
        return snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_summary(
        self, job_id: str, round_number: int
    ) -> RoundSummary:
        """Return an existing round summary or create a new one.

        Args:
            job_id: The federated job.
            round_number: The round to look up or create.

        Returns:
            RoundSummary (possibly newly created).
        """
        summaries = self._round_summaries.setdefault(job_id, [])
        for summary in summaries:
            if summary.round_number == round_number:
                return summary

        new_summary = RoundSummary(round_number=round_number)
        summaries.append(new_summary)
        return new_summary
