"""Participant registry adapter for federated learning node enrollment.

Manages the full lifecycle of participant enrollment: capability declaration,
health-check heartbeat monitoring, round assignment, approval workflows, and
contribution history. All operations are async and tenant-scoped.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Capability and status constants
# ---------------------------------------------------------------------------

PARTICIPANT_STATUSES = frozenset(
    {"pending_approval", "active", "idle", "dropped", "banned"}
)

REQUIRED_CAPABILITIES = frozenset({"compute", "dataset_size", "bandwidth_mbps"})


# ---------------------------------------------------------------------------
# Data classes (no ORM — registry state is in-memory + delegated to repo)
# ---------------------------------------------------------------------------


class ParticipantCapabilities:
    """Declared hardware and data capabilities for a participant node.

    Attributes:
        has_gpu: Whether the participant has GPU compute available.
        gpu_count: Number of GPUs available (0 if has_gpu is False).
        dataset_size: Number of training samples the participant contributes.
        bandwidth_mbps: Estimated upload bandwidth in megabits per second.
        compute_flops: Approximate compute capacity in GFLOPS (optional).
        supported_frameworks: List of ML framework names (e.g. ['torch', 'tensorflow']).
        custom_tags: Arbitrary key-value metadata for advanced matching.
    """

    def __init__(
        self,
        *,
        has_gpu: bool,
        gpu_count: int,
        dataset_size: int,
        bandwidth_mbps: float,
        compute_flops: float | None = None,
        supported_frameworks: list[str] | None = None,
        custom_tags: dict[str, str] | None = None,
    ) -> None:
        self.has_gpu = has_gpu
        self.gpu_count = max(0, gpu_count)
        self.dataset_size = max(0, dataset_size)
        self.bandwidth_mbps = max(0.0, bandwidth_mbps)
        self.compute_flops = compute_flops
        self.supported_frameworks: list[str] = supported_frameworks or []
        self.custom_tags: dict[str, str] = custom_tags or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize capabilities to a JSON-serialisable dict."""
        return {
            "has_gpu": self.has_gpu,
            "gpu_count": self.gpu_count,
            "dataset_size": self.dataset_size,
            "bandwidth_mbps": self.bandwidth_mbps,
            "compute_flops": self.compute_flops,
            "supported_frameworks": self.supported_frameworks,
            "custom_tags": self.custom_tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParticipantCapabilities":
        """Deserialise capabilities from a dict."""
        return cls(
            has_gpu=bool(data.get("has_gpu", False)),
            gpu_count=int(data.get("gpu_count", 0)),
            dataset_size=int(data.get("dataset_size", 0)),
            bandwidth_mbps=float(data.get("bandwidth_mbps", 0.0)),
            compute_flops=data.get("compute_flops"),
            supported_frameworks=list(data.get("supported_frameworks", [])),
            custom_tags=dict(data.get("custom_tags", {})),
        )


class ParticipantRecord:
    """In-registry participant record tracking liveness and history.

    Attributes:
        participant_id: Stable UUID string for this node.
        organization_id: Owning organization identifier.
        job_id: The federated job this participant belongs to.
        status: One of PARTICIPANT_STATUSES.
        capabilities: Declared hardware/data capabilities.
        last_heartbeat: UTC timestamp of the most recent ping.
        enrolled_at: UTC timestamp when registration was approved.
        rounds_assigned: List of round numbers this participant was assigned to.
        rounds_completed: List of round numbers successfully submitted.
        total_samples_contributed: Running sum of dataset_size across all rounds.
        approval_metadata: Extra data stored during the approval workflow.
    """

    def __init__(
        self,
        *,
        participant_id: str,
        organization_id: str,
        job_id: str,
        capabilities: ParticipantCapabilities,
        enrolled_at: datetime | None = None,
    ) -> None:
        self.participant_id = participant_id
        self.organization_id = organization_id
        self.job_id = job_id
        self.status: str = "pending_approval"
        self.capabilities = capabilities
        self.last_heartbeat: datetime | None = None
        self.enrolled_at: datetime | None = enrolled_at
        self.rounds_assigned: list[int] = []
        self.rounds_completed: list[int] = []
        self.total_samples_contributed: int = 0
        self.approval_metadata: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# ParticipantRegistry
# ---------------------------------------------------------------------------


class ParticipantRegistry:
    """Node enrollment registry for federated learning jobs.

    Manages participant registration, capability matching, heartbeat monitoring,
    round assignment, enrollment approval, and contribution history. State is
    stored in the injected repository; in-memory caches track heartbeat timing.

    Args:
        repository: Async repository providing save/load for participant records.
        heartbeat_timeout_seconds: Seconds without a heartbeat before a
            participant is considered dropped (default: 120).
        auto_approve: If True, registrations are auto-approved without a
            manual approval step (useful for testing/closed federations).
    """

    def __init__(
        self,
        repository: Any,
        heartbeat_timeout_seconds: int = 120,
        auto_approve: bool = False,
    ) -> None:
        self._repo = repository
        self._heartbeat_timeout = timedelta(seconds=heartbeat_timeout_seconds)
        self._auto_approve = auto_approve
        # participant_id -> ParticipantRecord (live cache updated by heartbeats)
        self._live_cache: dict[str, ParticipantRecord] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register_participant(
        self,
        *,
        job_id: str,
        organization_id: str,
        capabilities: dict[str, Any],
    ) -> str:
        """Enroll a new participant node for a federated job.

        Validates declared capabilities, creates a ParticipantRecord, and
        persists it. Returns the assigned participant_id.

        Args:
            job_id: The federated job this participant is joining.
            organization_id: Unique identifier for the enrolling organization.
            capabilities: Raw capability dict (validated against required keys).

        Returns:
            Assigned participant_id (UUID string).

        Raises:
            ValueError: If required capability keys are missing.
        """
        missing_keys = REQUIRED_CAPABILITIES - set(capabilities.keys())
        if missing_keys:
            raise ValueError(
                f"Missing required capability keys: {sorted(missing_keys)}"
            )

        parsed_capabilities = ParticipantCapabilities.from_dict(capabilities)
        participant_id = str(uuid.uuid4())

        now = datetime.now(tz=timezone.utc)
        record = ParticipantRecord(
            participant_id=participant_id,
            organization_id=organization_id,
            job_id=job_id,
            capabilities=parsed_capabilities,
        )

        if self._auto_approve:
            record.status = "idle"
            record.enrolled_at = now

        self._live_cache[participant_id] = record

        await self._repo.save_participant(
            participant_id=participant_id,
            job_id=job_id,
            organization_id=organization_id,
            status=record.status,
            capabilities=parsed_capabilities.to_dict(),
            enrolled_at=record.enrolled_at,
        )

        logger.info(
            "Registered participant %s for job %s (org=%s, status=%s)",
            participant_id,
            job_id,
            organization_id,
            record.status,
        )
        return participant_id

    # ------------------------------------------------------------------
    # Heartbeat / health
    # ------------------------------------------------------------------

    async def record_heartbeat(self, participant_id: str) -> None:
        """Record a liveness ping from a participant node.

        Updates last_heartbeat timestamp and transitions status from idle to
        active if necessary.

        Args:
            participant_id: The participant sending the heartbeat.

        Raises:
            KeyError: If participant_id is not found in the registry.
        """
        record = await self._get_or_load(participant_id)

        now = datetime.now(tz=timezone.utc)
        record.last_heartbeat = now

        if record.status == "idle":
            record.status = "active"

        await self._repo.update_heartbeat(participant_id=participant_id, timestamp=now)
        logger.debug("Heartbeat received from participant %s", participant_id)

    async def check_liveness(self, participant_id: str) -> bool:
        """Return True if the participant has heartbeated within the timeout window.

        Args:
            participant_id: The participant to check.

        Returns:
            True if alive, False if timed-out or never heartbeated.
        """
        record = await self._get_or_load(participant_id)
        if record.last_heartbeat is None:
            return False
        age = datetime.now(tz=timezone.utc) - record.last_heartbeat
        return age <= self._heartbeat_timeout

    async def sweep_dropped_participants(self, job_id: str) -> list[str]:
        """Scan all active participants for a job and mark timed-out nodes as dropped.

        Returns:
            List of participant_ids that were newly marked as dropped.
        """
        active_records = await self._repo.list_by_job_and_status(
            job_id=job_id, statuses=["active", "idle"]
        )
        dropped_ids: list[str] = []

        for raw in active_records:
            pid = raw["participant_id"]
            record = await self._get_or_load(pid)
            alive = await self.check_liveness(pid)
            if not alive:
                record.status = "dropped"
                await self._repo.update_status(participant_id=pid, status="dropped")
                dropped_ids.append(pid)
                logger.warning(
                    "Participant %s marked as dropped (heartbeat timeout)", pid
                )

        return dropped_ids

    # ------------------------------------------------------------------
    # Capability matching
    # ------------------------------------------------------------------

    async def find_eligible_participants(
        self,
        job_id: str,
        *,
        require_gpu: bool = False,
        min_dataset_size: int = 0,
        min_bandwidth_mbps: float = 0.0,
        required_framework: str | None = None,
    ) -> list[str]:
        """Return participant_ids that satisfy all specified capability requirements.

        Args:
            job_id: The job to filter participants for.
            require_gpu: If True, only return participants with GPU.
            min_dataset_size: Minimum number of training samples required.
            min_bandwidth_mbps: Minimum upload bandwidth required.
            required_framework: If set, participant must support this framework.

        Returns:
            List of matching participant_ids (active or idle only).
        """
        eligible: list[str] = []
        candidates = await self._repo.list_by_job_and_status(
            job_id=job_id, statuses=["active", "idle"]
        )

        for raw in candidates:
            pid = raw["participant_id"]
            caps = ParticipantCapabilities.from_dict(raw.get("capabilities", {}))

            if require_gpu and not caps.has_gpu:
                continue
            if caps.dataset_size < min_dataset_size:
                continue
            if caps.bandwidth_mbps < min_bandwidth_mbps:
                continue
            if required_framework and required_framework not in caps.supported_frameworks:
                continue

            eligible.append(pid)

        logger.info(
            "Found %d eligible participants for job %s", len(eligible), job_id
        )
        return eligible

    # ------------------------------------------------------------------
    # Round assignment
    # ------------------------------------------------------------------

    async def assign_to_round(
        self,
        job_id: str,
        round_number: int,
        participant_ids: list[str],
    ) -> None:
        """Record round assignment for a set of participants.

        Args:
            job_id: The federated job.
            round_number: The round these participants are assigned to.
            participant_ids: Participants selected for this round.
        """
        for pid in participant_ids:
            record = await self._get_or_load(pid)
            if round_number not in record.rounds_assigned:
                record.rounds_assigned.append(round_number)
            record.status = "active"

        await self._repo.bulk_assign_round(
            job_id=job_id,
            round_number=round_number,
            participant_ids=participant_ids,
        )

        logger.info(
            "Assigned %d participants to job %s round %d",
            len(participant_ids),
            job_id,
            round_number,
        )

    async def record_round_completion(
        self,
        participant_id: str,
        round_number: int,
        num_samples: int,
    ) -> None:
        """Record that a participant successfully submitted their round update.

        Args:
            participant_id: The completing participant.
            round_number: The round they completed.
            num_samples: Samples used in this round's training.
        """
        record = await self._get_or_load(participant_id)
        if round_number not in record.rounds_completed:
            record.rounds_completed.append(round_number)
        record.total_samples_contributed += num_samples
        record.status = "idle"

        await self._repo.record_round_completion(
            participant_id=participant_id,
            round_number=round_number,
            num_samples=num_samples,
        )

        logger.info(
            "Participant %s completed round %d (%d samples)",
            participant_id,
            round_number,
            num_samples,
        )

    # ------------------------------------------------------------------
    # Enrollment approval workflow
    # ------------------------------------------------------------------

    async def approve_enrollment(
        self,
        participant_id: str,
        *,
        approved_by: str,
        approval_notes: str | None = None,
    ) -> None:
        """Approve a pending enrollment, transitioning status to idle.

        Args:
            participant_id: The participant to approve.
            approved_by: Identity of the approver (user/system).
            approval_notes: Optional free-text notes attached to approval.

        Raises:
            ValueError: If participant is not in pending_approval status.
        """
        record = await self._get_or_load(participant_id)
        if record.status != "pending_approval":
            raise ValueError(
                f"Participant {participant_id} is in status '{record.status}', "
                "not 'pending_approval'"
            )

        now = datetime.now(tz=timezone.utc)
        record.status = "idle"
        record.enrolled_at = now
        record.approval_metadata = {
            "approved_by": approved_by,
            "approved_at": now.isoformat(),
            "notes": approval_notes,
        }

        await self._repo.update_status(
            participant_id=participant_id,
            status="idle",
            extra={"enrolled_at": now, "approval_metadata": record.approval_metadata},
        )

        logger.info(
            "Enrollment approved for participant %s by %s", participant_id, approved_by
        )

    async def reject_enrollment(
        self,
        participant_id: str,
        *,
        rejected_by: str,
        reason: str,
    ) -> None:
        """Reject a pending enrollment, transitioning status to banned.

        Args:
            participant_id: The participant to reject.
            rejected_by: Identity of the rejector.
            reason: Human-readable rejection reason.

        Raises:
            ValueError: If participant is not in pending_approval status.
        """
        record = await self._get_or_load(participant_id)
        if record.status != "pending_approval":
            raise ValueError(
                f"Participant {participant_id} is not pending approval (status={record.status})"
            )

        record.status = "banned"
        record.approval_metadata = {
            "rejected_by": rejected_by,
            "rejected_at": datetime.now(tz=timezone.utc).isoformat(),
            "reason": reason,
        }

        await self._repo.update_status(
            participant_id=participant_id,
            status="banned",
            extra={"approval_metadata": record.approval_metadata},
        )

        logger.info(
            "Enrollment rejected for participant %s by %s: %s",
            participant_id,
            rejected_by,
            reason,
        )

    # ------------------------------------------------------------------
    # History and contribution tracking
    # ------------------------------------------------------------------

    async def get_participant_history(
        self, participant_id: str
    ) -> dict[str, Any]:
        """Return a contribution history summary for a participant.

        Args:
            participant_id: The participant to query.

        Returns:
            Dict with keys: participant_id, rounds_assigned, rounds_completed,
            completion_rate, total_samples_contributed, status, enrolled_at.
        """
        record = await self._get_or_load(participant_id)
        assigned_count = len(record.rounds_assigned)
        completed_count = len(record.rounds_completed)
        completion_rate = (
            completed_count / assigned_count if assigned_count > 0 else 0.0
        )

        return {
            "participant_id": participant_id,
            "organization_id": record.organization_id,
            "job_id": record.job_id,
            "status": record.status,
            "enrolled_at": record.enrolled_at.isoformat() if record.enrolled_at else None,
            "rounds_assigned": sorted(record.rounds_assigned),
            "rounds_completed": sorted(record.rounds_completed),
            "completion_rate": round(completion_rate, 4),
            "total_samples_contributed": record.total_samples_contributed,
            "capabilities": record.capabilities.to_dict(),
        }

    async def get_job_participant_summary(self, job_id: str) -> dict[str, Any]:
        """Return a status summary for all participants in a job.

        Args:
            job_id: The job to summarize.

        Returns:
            Dict with counts per status and a list of participant summaries.
        """
        all_raw = await self._repo.list_by_job(job_id=job_id)
        status_counts: dict[str, int] = {}
        summaries: list[dict[str, Any]] = []

        for raw in all_raw:
            pid = raw["participant_id"]
            status = raw.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            summaries.append(
                {
                    "participant_id": pid,
                    "organization_id": raw.get("organization_id"),
                    "status": status,
                    "last_heartbeat": raw.get("last_heartbeat"),
                }
            )

        return {
            "job_id": job_id,
            "total_participants": len(all_raw),
            "status_counts": status_counts,
            "participants": summaries,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_or_load(self, participant_id: str) -> ParticipantRecord:
        """Return cached record, or load from repository if not in cache.

        Args:
            participant_id: The participant to retrieve.

        Returns:
            Live ParticipantRecord.

        Raises:
            KeyError: If participant_id does not exist in the repository.
        """
        if participant_id in self._live_cache:
            return self._live_cache[participant_id]

        raw = await self._repo.get_participant(participant_id)
        if raw is None:
            raise KeyError(f"Participant {participant_id} not found in registry")

        caps = ParticipantCapabilities.from_dict(raw.get("capabilities", {}))
        record = ParticipantRecord(
            participant_id=participant_id,
            organization_id=raw.get("organization_id", ""),
            job_id=raw.get("job_id", ""),
            capabilities=caps,
        )
        record.status = raw.get("status", "pending_approval")
        record.rounds_assigned = list(raw.get("rounds_assigned", []))
        record.rounds_completed = list(raw.get("rounds_completed", []))
        record.total_samples_contributed = int(
            raw.get("total_samples_contributed", 0)
        )
        raw_hb = raw.get("last_heartbeat")
        if isinstance(raw_hb, datetime):
            record.last_heartbeat = raw_hb
        enrolled_raw = raw.get("enrolled_at")
        if isinstance(enrolled_raw, datetime):
            record.enrolled_at = enrolled_raw

        self._live_cache[participant_id] = record
        return record
