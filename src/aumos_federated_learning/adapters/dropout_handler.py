"""Dropout handler adapter for federated learning straggler tolerance.

Detects timed-out participants, enables partial aggregation from available
updates, enforces minimum quorum requirements, manages straggler replacement,
and supports re-synchronisation of participants that re-join after dropping.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ROUND_TIMEOUT_SECONDS = 3600       # 1 hour
DEFAULT_EXTENSION_SECONDS = 600            # 10 minute extension for stragglers
DEFAULT_MIN_QUORUM_FRACTION = 0.5          # 50% of assigned participants
MAX_EXTENSION_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RoundDropoutState:
    """Tracks dropout state for a single training round.

    Attributes:
        job_id: The federated job.
        round_number: The round being tracked.
        assigned_participant_ids: All participants scheduled for this round.
        submitted_participant_ids: Participants who submitted updates on time.
        dropped_participant_ids: Participants confirmed as dropped/timed-out.
        straggler_participant_ids: Participants still pending (in extension window).
        deadline: UTC datetime by which updates must be submitted.
        extended_deadline: Optional extended deadline after straggler extension.
        extension_count: Number of times the deadline has been extended.
        partial_aggregation_triggered: Whether aggregation ran without full quorum.
        created_at: When this round state was created.
    """

    job_id: str
    round_number: int
    assigned_participant_ids: list[str]
    submitted_participant_ids: list[str] = field(default_factory=list)
    dropped_participant_ids: list[str] = field(default_factory=list)
    straggler_participant_ids: list[str] = field(default_factory=list)
    deadline: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    extended_deadline: datetime | None = None
    extension_count: int = 0
    partial_aggregation_triggered: bool = False
    created_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    @property
    def pending_participant_ids(self) -> list[str]:
        """Participants assigned but not yet submitted or dropped."""
        submitted_set = set(self.submitted_participant_ids)
        dropped_set = set(self.dropped_participant_ids)
        return [
            pid
            for pid in self.assigned_participant_ids
            if pid not in submitted_set and pid not in dropped_set
        ]

    @property
    def submission_rate(self) -> float:
        """Fraction of assigned participants who submitted."""
        if not self.assigned_participant_ids:
            return 0.0
        return len(self.submitted_participant_ids) / len(self.assigned_participant_ids)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "job_id": self.job_id,
            "round_number": self.round_number,
            "assigned": len(self.assigned_participant_ids),
            "submitted": len(self.submitted_participant_ids),
            "dropped": len(self.dropped_participant_ids),
            "stragglers": len(self.straggler_participant_ids),
            "pending": len(self.pending_participant_ids),
            "submission_rate": round(self.submission_rate, 4),
            "deadline": self.deadline.isoformat(),
            "extended_deadline": (
                self.extended_deadline.isoformat()
                if self.extended_deadline
                else None
            ),
            "extension_count": self.extension_count,
            "partial_aggregation_triggered": self.partial_aggregation_triggered,
        }


# ---------------------------------------------------------------------------
# DropoutHandler
# ---------------------------------------------------------------------------


class DropoutHandler:
    """Straggler tolerance and churn recovery for federated training rounds.

    Manages:
    - Per-round deadline tracking with extension support
    - Timeout detection for slow participants
    - Partial aggregation when a minimum quorum is met but not full participation
    - Straggler replacement policy (substituting an idle participant)
    - Re-synchronisation of previously dropped participants for future rounds
    - Dropout statistics accumulation for alerting

    Args:
        participant_registry: Async registry exposing find_eligible_participants,
            get_job_participant_summary, and assign_to_round methods.
        event_publisher: Async publisher for dropout alert events (may be None).
        round_timeout_seconds: Default maximum wait time per round.
        extension_seconds: Time added when extending deadline for stragglers.
        min_quorum_fraction: Minimum fraction of assigned participants required
            to proceed with partial aggregation.
    """

    def __init__(
        self,
        participant_registry: Any,
        event_publisher: Any | None = None,
        round_timeout_seconds: int = DEFAULT_ROUND_TIMEOUT_SECONDS,
        extension_seconds: int = DEFAULT_EXTENSION_SECONDS,
        min_quorum_fraction: float = DEFAULT_MIN_QUORUM_FRACTION,
    ) -> None:
        self._registry = participant_registry
        self._publisher = event_publisher
        self._round_timeout = round_timeout_seconds
        self._extension_seconds = extension_seconds
        self._min_quorum_fraction = min_quorum_fraction

        # (job_id, round_number) -> RoundDropoutState
        self._round_states: dict[tuple[str, int], RoundDropoutState] = {}

        # job_id -> cumulative dropout stats
        self._dropout_stats: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Round registration
    # ------------------------------------------------------------------

    def register_round(
        self,
        job_id: str,
        round_number: int,
        assigned_participant_ids: list[str],
        deadline: datetime | None = None,
    ) -> RoundDropoutState:
        """Register participants and deadline for a training round.

        Args:
            job_id: The federated job.
            round_number: The round being registered.
            assigned_participant_ids: Participants scheduled for this round.
            deadline: Hard deadline for submission (defaults to now + timeout).

        Returns:
            Newly created RoundDropoutState.
        """
        if deadline is None:
            deadline = datetime.now(tz=timezone.utc) + timedelta(
                seconds=self._round_timeout
            )

        state = RoundDropoutState(
            job_id=job_id,
            round_number=round_number,
            assigned_participant_ids=list(assigned_participant_ids),
            deadline=deadline,
        )
        self._round_states[(job_id, round_number)] = state

        logger.info(
            "Registered dropout tracking for job %s round %d "
            "(%d participants, deadline=%s)",
            job_id,
            round_number,
            len(assigned_participant_ids),
            deadline.isoformat(),
        )
        return state

    # ------------------------------------------------------------------
    # Submission recording
    # ------------------------------------------------------------------

    def record_submission(
        self, job_id: str, round_number: int, participant_id: str
    ) -> None:
        """Mark a participant as having submitted their update for the round.

        Args:
            job_id: The federated job.
            round_number: The round the update belongs to.
            participant_id: The participant who submitted.

        Raises:
            KeyError: If no dropout state was registered for this round.
        """
        state = self._get_state(job_id, round_number)
        if participant_id not in state.submitted_participant_ids:
            state.submitted_participant_ids.append(participant_id)
        # Remove from straggler list if they eventually submitted
        if participant_id in state.straggler_participant_ids:
            state.straggler_participant_ids.remove(participant_id)

        logger.debug(
            "Participant %s submitted for job %s round %d (%d/%d submitted)",
            participant_id,
            job_id,
            round_number,
            len(state.submitted_participant_ids),
            len(state.assigned_participant_ids),
        )

    # ------------------------------------------------------------------
    # Timeout detection
    # ------------------------------------------------------------------

    async def detect_timeouts(
        self, job_id: str, round_number: int
    ) -> list[str]:
        """Identify participants who have missed the round deadline.

        Participants pending after the deadline are flagged as stragglers
        or dropped depending on whether any extension remains.

        Args:
            job_id: The federated job.
            round_number: The round to check.

        Returns:
            List of participant_ids that are confirmed timed-out (dropped).
        """
        state = self._get_state(job_id, round_number)
        now = datetime.now(tz=timezone.utc)
        effective_deadline = state.extended_deadline or state.deadline

        if now <= effective_deadline:
            # Still within window — check liveness for early dropout detection
            timed_out: list[str] = []
            for pid in state.pending_participant_ids:
                alive = await self._registry.check_liveness(pid)
                if not alive:
                    timed_out.append(pid)
            return timed_out

        # Past deadline — all pending are timed out
        timed_out = list(state.pending_participant_ids)
        for pid in timed_out:
            if pid not in state.dropped_participant_ids:
                state.dropped_participant_ids.append(pid)

        if timed_out:
            logger.warning(
                "Timeout detected for job %s round %d: %d participants dropped: %s",
                job_id,
                round_number,
                len(timed_out),
                timed_out,
            )
            await self._emit_dropout_alert(job_id, round_number, timed_out)
            self._update_dropout_stats(job_id, dropped_count=len(timed_out))

        return timed_out

    # ------------------------------------------------------------------
    # Quorum enforcement
    # ------------------------------------------------------------------

    def check_quorum(self, job_id: str, round_number: int) -> bool:
        """Return True if the minimum quorum fraction has been met.

        Args:
            job_id: The federated job.
            round_number: The round to check.

        Returns:
            True if enough participants have submitted to proceed.
        """
        state = self._get_state(job_id, round_number)
        min_required = max(
            1,
            int(self._min_quorum_fraction * len(state.assigned_participant_ids)),
        )
        quorum_met = len(state.submitted_participant_ids) >= min_required

        logger.info(
            "Quorum check for job %s round %d: %d/%d submitted (min=%d) — %s",
            job_id,
            round_number,
            len(state.submitted_participant_ids),
            len(state.assigned_participant_ids),
            min_required,
            "MET" if quorum_met else "NOT MET",
        )
        return quorum_met

    def enforce_minimum_quorum(
        self,
        job_id: str,
        round_number: int,
        min_participants: int,
    ) -> None:
        """Raise an error if the absolute minimum participant count is not met.

        Args:
            job_id: The federated job.
            round_number: The round to enforce on.
            min_participants: Absolute minimum required submission count.

        Raises:
            ValueError: If submitted count is below min_participants.
        """
        state = self._get_state(job_id, round_number)
        submitted = len(state.submitted_participant_ids)
        if submitted < min_participants:
            raise ValueError(
                f"Job {job_id} round {round_number}: only {submitted}/{min_participants} "
                "participants submitted — cannot proceed with aggregation"
            )

    # ------------------------------------------------------------------
    # Partial aggregation
    # ------------------------------------------------------------------

    def get_available_updates(
        self, job_id: str, round_number: int
    ) -> list[str]:
        """Return participant IDs that have submitted updates for partial aggregation.

        Args:
            job_id: The federated job.
            round_number: The round to query.

        Returns:
            List of participant_ids with submitted updates.
        """
        state = self._get_state(job_id, round_number)
        return list(state.submitted_participant_ids)

    def trigger_partial_aggregation(
        self, job_id: str, round_number: int
    ) -> list[str]:
        """Mark the round as using partial aggregation and return available participants.

        Should be called after quorum is confirmed but full participation is not reached.

        Args:
            job_id: The federated job.
            round_number: The round to partially aggregate.

        Returns:
            Participant IDs whose updates should be included.

        Raises:
            ValueError: If quorum is not met.
        """
        if not self.check_quorum(job_id, round_number):
            raise ValueError(
                f"Cannot trigger partial aggregation: quorum not met "
                f"for job {job_id} round {round_number}"
            )

        state = self._get_state(job_id, round_number)
        state.partial_aggregation_triggered = True

        logger.info(
            "Partial aggregation triggered for job %s round %d "
            "(%d/%d participants)",
            job_id,
            round_number,
            len(state.submitted_participant_ids),
            len(state.assigned_participant_ids),
        )
        return list(state.submitted_participant_ids)

    # ------------------------------------------------------------------
    # Straggler extension
    # ------------------------------------------------------------------

    async def extend_round_deadline(
        self,
        job_id: str,
        round_number: int,
    ) -> datetime | None:
        """Extend the round deadline to give stragglers more time.

        Applies up to MAX_EXTENSION_ATTEMPTS extensions. After the limit is
        reached, no further extension is granted.

        Args:
            job_id: The federated job.
            round_number: The round to extend.

        Returns:
            The new extended deadline, or None if the extension limit is reached.
        """
        state = self._get_state(job_id, round_number)

        if state.extension_count >= MAX_EXTENSION_ATTEMPTS:
            logger.warning(
                "Extension limit (%d) reached for job %s round %d — no further extension",
                MAX_EXTENSION_ATTEMPTS,
                job_id,
                round_number,
            )
            return None

        base = state.extended_deadline or state.deadline
        new_deadline = base + timedelta(seconds=self._extension_seconds)
        state.extended_deadline = new_deadline
        state.extension_count += 1

        pending = state.pending_participant_ids
        state.straggler_participant_ids = [
            pid for pid in pending if pid not in state.straggler_participant_ids
        ]

        logger.info(
            "Extended deadline for job %s round %d to %s "
            "(%d stragglers: %s)",
            job_id,
            round_number,
            new_deadline.isoformat(),
            len(state.straggler_participant_ids),
            state.straggler_participant_ids,
        )
        return new_deadline

    # ------------------------------------------------------------------
    # Straggler replacement
    # ------------------------------------------------------------------

    async def replace_stragglers(
        self,
        job_id: str,
        round_number: int,
    ) -> list[str]:
        """Attempt to recruit idle participants to replace stragglers.

        Queries the participant registry for eligible idle participants and
        assigns them to the current round as replacements.

        Args:
            job_id: The federated job.
            round_number: The round needing replacement participants.

        Returns:
            List of participant_ids that were recruited as replacements.
        """
        state = self._get_state(job_id, round_number)
        straggler_count = len(state.straggler_participant_ids)
        if straggler_count == 0:
            return []

        excluded = set(
            state.assigned_participant_ids
            + state.dropped_participant_ids
        )
        eligible = await self._registry.find_eligible_participants(job_id=job_id)
        replacements = [pid for pid in eligible if pid not in excluded][:straggler_count]

        if replacements:
            await self._registry.assign_to_round(
                job_id=job_id,
                round_number=round_number,
                participant_ids=replacements,
            )
            state.assigned_participant_ids.extend(replacements)
            logger.info(
                "Recruited %d replacement participants for job %s round %d: %s",
                len(replacements),
                job_id,
                round_number,
                replacements,
            )
        else:
            logger.warning(
                "No idle replacements available for job %s round %d "
                "(%d stragglers unresolved)",
                job_id,
                round_number,
                straggler_count,
            )

        return replacements

    # ------------------------------------------------------------------
    # Churn recovery
    # ------------------------------------------------------------------

    async def resync_dropped_participant(
        self,
        job_id: str,
        participant_id: str,
        next_round_number: int,
    ) -> bool:
        """Re-enroll a previously dropped participant for a future round.

        Verifies liveness, updates registry status, and adds the participant
        to the assignment list for the specified next round.

        Args:
            job_id: The federated job.
            participant_id: The participant requesting re-synchronisation.
            next_round_number: The round to assign the participant to.

        Returns:
            True if re-sync succeeded, False if the participant is still unreachable.
        """
        alive = await self._registry.check_liveness(participant_id)
        if not alive:
            logger.warning(
                "Cannot resync participant %s for job %s — still unreachable",
                participant_id,
                job_id,
            )
            return False

        # Register participant for the next round
        state_key = (job_id, next_round_number)
        if state_key in self._round_states:
            state = self._round_states[state_key]
            if participant_id not in state.assigned_participant_ids:
                state.assigned_participant_ids.append(participant_id)
                # Remove from dropped lists in this and previous round states
                for round_state in self._round_states.values():
                    if round_state.job_id == job_id:
                        if participant_id in round_state.dropped_participant_ids:
                            round_state.dropped_participant_ids.remove(participant_id)

        logger.info(
            "Re-synced participant %s for job %s starting at round %d",
            participant_id,
            job_id,
            next_round_number,
        )
        return True

    # ------------------------------------------------------------------
    # Statistics and alerting
    # ------------------------------------------------------------------

    def get_dropout_statistics(self, job_id: str) -> dict[str, Any]:
        """Return cumulative dropout statistics for a job.

        Args:
            job_id: The federated job.

        Returns:
            Dict with total_rounds_tracked, total_dropouts, total_extensions,
            partial_aggregation_count, per_round_summary.
        """
        per_round: list[dict[str, Any]] = []
        total_dropouts = 0
        total_extensions = 0
        partial_count = 0

        for (jid, _round_number), state in sorted(
            self._round_states.items(), key=lambda kv: kv[0][1]
        ):
            if jid != job_id:
                continue
            total_dropouts += len(state.dropped_participant_ids)
            total_extensions += state.extension_count
            if state.partial_aggregation_triggered:
                partial_count += 1
            per_round.append(state.to_dict())

        cumulative = self._dropout_stats.get(job_id, {})

        return {
            "job_id": job_id,
            "total_rounds_tracked": len(per_round),
            "total_dropouts": total_dropouts + cumulative.get("total_dropouts", 0),
            "total_extensions": total_extensions,
            "partial_aggregation_count": partial_count,
            "per_round_summary": per_round,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self, job_id: str, round_number: int) -> RoundDropoutState:
        """Retrieve round dropout state or raise a descriptive error.

        Args:
            job_id: The federated job.
            round_number: The round to look up.

        Returns:
            RoundDropoutState for the specified round.

        Raises:
            KeyError: If the round was not registered.
        """
        key = (job_id, round_number)
        state = self._round_states.get(key)
        if state is None:
            raise KeyError(
                f"No dropout state registered for job {job_id} round {round_number}. "
                "Call register_round() first."
            )
        return state

    def _update_dropout_stats(self, job_id: str, *, dropped_count: int) -> None:
        """Accumulate dropout statistics for a job.

        Args:
            job_id: The federated job.
            dropped_count: Number of new dropouts to record.
        """
        if job_id not in self._dropout_stats:
            self._dropout_stats[job_id] = {"total_dropouts": 0}
        self._dropout_stats[job_id]["total_dropouts"] += dropped_count

    async def _emit_dropout_alert(
        self,
        job_id: str,
        round_number: int,
        dropped_participants: list[str],
    ) -> None:
        """Publish a dropout alert event if an event publisher is configured.

        Args:
            job_id: The federated job.
            round_number: The round where dropout occurred.
            dropped_participants: Participant IDs that dropped out.
        """
        if self._publisher is None:
            return

        event = {
            "event_type": "fl.participant.dropout",
            "job_id": job_id,
            "round_number": round_number,
            "dropped_count": len(dropped_participants),
            "participant_ids": dropped_participants,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

        try:
            await self._publisher.publish(topic="fl-dropout-alerts", payload=event)
        except Exception as exc:
            logger.warning("Failed to emit dropout alert: %s", exc)
