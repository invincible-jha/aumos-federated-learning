"""Federated model versioner adapter — training round checkpoint management.

Tracks round numbers, stores global model checkpoints to object storage
(MinIO/S3), computes weight deltas between rounds, manages version history,
supports rollback, and prunes old checkpoints according to a configurable
retention policy.
"""

from __future__ import annotations

import io
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT_BUCKET = "aumos-fl-checkpoints"
DEFAULT_MAX_CHECKPOINTS_RETAINED = 10


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ModelVersion:
    """Metadata for a single global model checkpoint.

    Attributes:
        version_id: Unique UUID string for this version.
        job_id: The federated job this version belongs to.
        round_number: The training round that produced this checkpoint.
        object_key: Storage key (MinIO/S3) where the weights are stored.
        num_parameters: Total count of scalar parameters in the model.
        size_bytes: On-disk size of the serialized checkpoint.
        metrics: Dict of training metrics captured alongside the checkpoint.
        created_at: UTC timestamp when the checkpoint was saved.
        parent_version_id: Version ID of the previous round (None for round 1).
    """

    version_id: str
    job_id: str
    round_number: int
    object_key: str
    num_parameters: int
    size_bytes: int
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    parent_version_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "version_id": self.version_id,
            "job_id": self.job_id,
            "round_number": self.round_number,
            "object_key": self.object_key,
            "num_parameters": self.num_parameters,
            "size_bytes": self.size_bytes,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
            "parent_version_id": self.parent_version_id,
        }


# ---------------------------------------------------------------------------
# FederatedModelVersioner
# ---------------------------------------------------------------------------


class FederatedModelVersioner:
    """Manages global model checkpoints across federated training rounds.

    Responsibilities:
    - Assign and track round numbers per job
    - Store serialized model weights in object storage (MinIO/S3 compatible)
    - Compute parameter deltas (delta weights) between consecutive rounds
    - Maintain a versioned history with rich metadata
    - Support rollback to any previous checkpoint
    - Prune old checkpoints beyond the retention window

    Args:
        object_storage: Async object storage client providing put_object /
            get_object / delete_object / list_objects methods.
        metadata_repository: Async repository for persisting ModelVersion records.
        bucket_name: Object storage bucket for checkpoint storage.
        max_checkpoints_retained: Maximum checkpoints to keep per job before
            pruning the oldest.
    """

    def __init__(
        self,
        object_storage: Any,
        metadata_repository: Any,
        bucket_name: str = DEFAULT_CHECKPOINT_BUCKET,
        max_checkpoints_retained: int = DEFAULT_MAX_CHECKPOINTS_RETAINED,
    ) -> None:
        self._storage = object_storage
        self._meta_repo = metadata_repository
        self._bucket = bucket_name
        self._max_retained = max_checkpoints_retained

        # job_id -> current round number (in-memory cache)
        self._current_rounds: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Round tracking
    # ------------------------------------------------------------------

    async def get_current_round(self, job_id: str) -> int:
        """Return the most recently completed round number for a job.

        Args:
            job_id: The federated job to query.

        Returns:
            Round number (0 if no rounds have been completed yet).
        """
        if job_id in self._current_rounds:
            return self._current_rounds[job_id]

        latest = await self._meta_repo.get_latest_version(job_id=job_id)
        round_number: int = latest.round_number if latest is not None else 0
        self._current_rounds[job_id] = round_number
        return round_number

    async def advance_round(self, job_id: str) -> int:
        """Increment and return the next round number for a job.

        Args:
            job_id: The federated job advancing to the next round.

        Returns:
            New current round number.
        """
        current = await self.get_current_round(job_id)
        next_round = current + 1
        self._current_rounds[job_id] = next_round
        logger.info("Job %s advanced to round %d", job_id, next_round)
        return next_round

    # ------------------------------------------------------------------
    # Checkpoint storage
    # ------------------------------------------------------------------

    async def save_checkpoint(
        self,
        job_id: str,
        round_number: int,
        parameters: list[np.ndarray[Any, Any]],
        metrics: dict[str, Any] | None = None,
    ) -> ModelVersion:
        """Serialize and store a global model checkpoint in object storage.

        Args:
            job_id: The federated job this checkpoint belongs to.
            round_number: The completed training round.
            parameters: List of NumPy model weight arrays.
            metrics: Optional training metrics for this round.

        Returns:
            ModelVersion record with storage metadata.
        """
        version_id = str(uuid.uuid4())
        object_key = f"{job_id}/round_{round_number:05d}/{version_id}.npz"

        serialized_bytes = self._serialize_weights(parameters)
        num_parameters = int(sum(arr.size for arr in parameters))

        await self._storage.put_object(
            bucket=self._bucket,
            key=object_key,
            data=serialized_bytes,
            content_type="application/octet-stream",
        )

        # Find parent version
        parent_version = await self._meta_repo.get_latest_version(job_id=job_id)
        parent_version_id: str | None = (
            parent_version.version_id if parent_version is not None else None
        )

        version = ModelVersion(
            version_id=version_id,
            job_id=job_id,
            round_number=round_number,
            object_key=object_key,
            num_parameters=num_parameters,
            size_bytes=len(serialized_bytes),
            metrics=metrics or {},
            parent_version_id=parent_version_id,
        )

        await self._meta_repo.save_version(version)
        self._current_rounds[job_id] = round_number

        logger.info(
            "Saved checkpoint version %s for job %s round %d "
            "(%d parameters, %d bytes, key=%s)",
            version_id,
            job_id,
            round_number,
            num_parameters,
            len(serialized_bytes),
            object_key,
        )

        # Apply pruning policy after save
        await self._apply_pruning(job_id)

        return version

    async def load_checkpoint(
        self,
        job_id: str,
        round_number: int | None = None,
        version_id: str | None = None,
    ) -> tuple[list[np.ndarray[Any, Any]], ModelVersion]:
        """Load model weights from a checkpoint.

        Exactly one of round_number or version_id must be provided.
        If neither, returns the latest checkpoint.

        Args:
            job_id: The federated job to load from.
            round_number: Load the checkpoint for this specific round.
            version_id: Load a specific version by UUID.

        Returns:
            Tuple of (parameter_arrays, ModelVersion).

        Raises:
            ValueError: If no matching checkpoint is found.
        """
        if version_id is not None:
            version = await self._meta_repo.get_version(version_id=version_id)
        elif round_number is not None:
            version = await self._meta_repo.get_version_for_round(
                job_id=job_id, round_number=round_number
            )
        else:
            version = await self._meta_repo.get_latest_version(job_id=job_id)

        if version is None:
            identifier = version_id or round_number or "latest"
            raise ValueError(
                f"No checkpoint found for job {job_id} ({identifier})"
            )

        raw_bytes = await self._storage.get_object(
            bucket=self._bucket, key=version.object_key
        )
        parameters = self._deserialize_weights(raw_bytes)

        logger.info(
            "Loaded checkpoint version %s for job %s round %d",
            version.version_id,
            job_id,
            version.round_number,
        )
        return parameters, version

    # ------------------------------------------------------------------
    # Delta computation
    # ------------------------------------------------------------------

    async def compute_round_delta(
        self,
        job_id: str,
        from_round: int,
        to_round: int,
    ) -> list[np.ndarray[Any, Any]]:
        """Compute the parameter delta (diff) between two model versions.

        The delta is computed as: params[to_round] - params[from_round]
        for each layer.

        Args:
            job_id: The federated job.
            from_round: The earlier (base) round number.
            to_round: The later (target) round number.

        Returns:
            List of NumPy arrays representing the weight differences.

        Raises:
            ValueError: If either round is missing or the layer counts differ.
        """
        base_params, base_version = await self.load_checkpoint(
            job_id=job_id, round_number=from_round
        )
        target_params, target_version = await self.load_checkpoint(
            job_id=job_id, round_number=to_round
        )

        if len(base_params) != len(target_params):
            raise ValueError(
                f"Layer count mismatch: round {from_round} has {len(base_params)} layers, "
                f"round {to_round} has {len(target_params)} layers"
            )

        delta: list[np.ndarray[Any, Any]] = []
        for base_layer, target_layer in zip(base_params, target_params, strict=True):
            if base_layer.shape != target_layer.shape:
                raise ValueError(
                    f"Shape mismatch: {base_layer.shape} vs {target_layer.shape}"
                )
            delta.append((target_layer - base_layer).astype(np.float64))

        logger.info(
            "Computed delta for job %s rounds %d -> %d (%d layers)",
            job_id,
            from_round,
            to_round,
            len(delta),
        )
        return delta

    def compute_delta_norm(
        self, delta: list[np.ndarray[Any, Any]]
    ) -> float:
        """Compute the global L2 norm of a weight delta.

        Args:
            delta: List of per-layer delta arrays.

        Returns:
            Scalar L2 norm of the flattened concatenated delta.
        """
        flat = np.concatenate([layer.flatten() for layer in delta])
        return float(np.linalg.norm(flat))

    # ------------------------------------------------------------------
    # Version history
    # ------------------------------------------------------------------

    async def get_version_history(
        self,
        job_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return version metadata for all checkpoints of a job.

        Args:
            job_id: The federated job.
            limit: Maximum number of versions to return (most recent first).

        Returns:
            List of version metadata dicts ordered by round descending.
        """
        versions = await self._meta_repo.list_versions(
            job_id=job_id, limit=limit, order="desc"
        )
        return [v.to_dict() for v in versions]

    async def compare_rounds(
        self,
        job_id: str,
        round_a: int,
        round_b: int,
    ) -> dict[str, Any]:
        """Compare two model checkpoints and return summary statistics.

        Args:
            job_id: The federated job.
            round_a: First round to compare.
            round_b: Second round to compare.

        Returns:
            Dict with keys: delta_l2_norm, layer_count, size_change_bytes,
            from_round, to_round, metrics_comparison.
        """
        _, version_a = await self.load_checkpoint(job_id=job_id, round_number=round_a)
        _, version_b = await self.load_checkpoint(job_id=job_id, round_number=round_b)

        delta = await self.compute_round_delta(
            job_id=job_id, from_round=round_a, to_round=round_b
        )
        l2_norm = self.compute_delta_norm(delta)

        return {
            "from_round": round_a,
            "to_round": round_b,
            "delta_l2_norm": round(l2_norm, 6),
            "layer_count": len(delta),
            "size_change_bytes": version_b.size_bytes - version_a.size_bytes,
            "metrics_a": version_a.metrics,
            "metrics_b": version_b.metrics,
        }

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    async def rollback_to_round(
        self,
        job_id: str,
        target_round: int,
    ) -> ModelVersion:
        """Restore the global model to the state at a previous round.

        Loads the checkpoint from target_round, saves it as a new version
        tagged as a rollback, and updates the current round tracker.

        Args:
            job_id: The federated job.
            target_round: The round to roll back to.

        Returns:
            The new ModelVersion created for the rollback checkpoint.

        Raises:
            ValueError: If target_round checkpoint is not found.
        """
        parameters, source_version = await self.load_checkpoint(
            job_id=job_id, round_number=target_round
        )

        current_round = await self.get_current_round(job_id)
        rollback_round = current_round + 1

        rollback_metrics = {
            **source_version.metrics,
            "rollback": True,
            "rolled_back_from_round": current_round,
            "rolled_back_to_round": target_round,
        }

        new_version = await self.save_checkpoint(
            job_id=job_id,
            round_number=rollback_round,
            parameters=parameters,
            metrics=rollback_metrics,
        )

        logger.warning(
            "Rolled back job %s from round %d to round %d (new version %s)",
            job_id,
            current_round,
            target_round,
            new_version.version_id,
        )
        return new_version

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    async def prune_old_checkpoints(
        self,
        job_id: str,
        keep_latest: int | None = None,
    ) -> int:
        """Delete old checkpoint objects from storage beyond the retention limit.

        Args:
            job_id: The federated job to prune.
            keep_latest: Override the default retention count.

        Returns:
            Number of checkpoints deleted.
        """
        retain = keep_latest if keep_latest is not None else self._max_retained
        all_versions = await self._meta_repo.list_versions(
            job_id=job_id, limit=10_000, order="asc"
        )

        to_delete = all_versions[: max(0, len(all_versions) - retain)]
        deleted_count = 0

        for version in to_delete:
            try:
                await self._storage.delete_object(
                    bucket=self._bucket, key=version.object_key
                )
                await self._meta_repo.delete_version(version_id=version.version_id)
                deleted_count += 1
                logger.info(
                    "Pruned checkpoint version %s (job=%s round=%d)",
                    version.version_id,
                    job_id,
                    version.round_number,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to prune version %s: %s", version.version_id, exc
                )

        return deleted_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _apply_pruning(self, job_id: str) -> None:
        """Automatically run pruning after a new checkpoint is saved."""
        all_versions = await self._meta_repo.list_versions(
            job_id=job_id, limit=10_000, order="asc"
        )
        if len(all_versions) > self._max_retained:
            await self.prune_old_checkpoints(job_id=job_id)

    @staticmethod
    def _serialize_weights(parameters: list[np.ndarray[Any, Any]]) -> bytes:
        """Serialize model weights to NumPy NPZ binary format.

        Args:
            parameters: List of weight arrays.

        Returns:
            Compressed NPZ bytes.
        """
        buffer = io.BytesIO()
        arrays = {f"layer_{i}": arr for i, arr in enumerate(parameters)}
        np.savez_compressed(buffer, **arrays)
        return buffer.getvalue()

    @staticmethod
    def _deserialize_weights(data: bytes) -> list[np.ndarray[Any, Any]]:
        """Deserialize model weights from NPZ binary format.

        Args:
            data: Compressed NPZ bytes.

        Returns:
            List of weight arrays in original layer order.
        """
        buffer = io.BytesIO(data)
        loaded = np.load(buffer, allow_pickle=False)
        sorted_keys = sorted(loaded.files, key=lambda k: int(k.split("_")[1]))
        return [loaded[key] for key in sorted_keys]
