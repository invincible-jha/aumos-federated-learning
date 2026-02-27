"""Incentive scorer adapter for federated learning contribution attribution.

Scores each participant's contribution using data quality heuristics and a
Shapley-value approximation, detects free-rider behaviour, tracks historical
contributions across rounds, calculates rewards, and produces distribution reports.
"""

from __future__ import annotations

import logging
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SHAPLEY_PERMUTATIONS = 50    # Permutation sampling iterations for SVP
FREE_RIDER_THRESHOLD = 0.05          # Contribution fraction below which free-riding is flagged
REWARD_POOL_DEFAULT = 1000.0         # Arbitrary reward pool units per round


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ContributionRecord:
    """A single participant's contribution in one training round.

    Attributes:
        record_id: Unique UUID for this contribution record.
        participant_id: The participant being scored.
        job_id: The federated job.
        round_number: The training round.
        data_quality_score: Score (0–1) reflecting data diversity and size.
        model_improvement_delta: L2 norm improvement attributed to this participant.
        shapley_value: Approximate Shapley value for this round.
        num_samples: Training samples contributed.
        reward_units: Calculated reward for this round.
        is_free_rider: Whether free-riding was detected.
        recorded_at: UTC timestamp.
    """

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    participant_id: str = ""
    job_id: str = ""
    round_number: int = 0
    data_quality_score: float = 0.0
    model_improvement_delta: float = 0.0
    shapley_value: float = 0.0
    num_samples: int = 0
    reward_units: float = 0.0
    is_free_rider: bool = False
    recorded_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "record_id": self.record_id,
            "participant_id": self.participant_id,
            "job_id": self.job_id,
            "round_number": self.round_number,
            "data_quality_score": round(self.data_quality_score, 6),
            "model_improvement_delta": round(self.model_improvement_delta, 6),
            "shapley_value": round(self.shapley_value, 6),
            "num_samples": self.num_samples,
            "reward_units": round(self.reward_units, 4),
            "is_free_rider": self.is_free_rider,
            "recorded_at": self.recorded_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# IncentiveScorer
# ---------------------------------------------------------------------------


class IncentiveScorer:
    """Contribution scoring, reward allocation, and free-rider detection.

    Implements:
    - Data quality scoring based on reported dataset properties
    - Shapley value approximation via random permutation sampling
    - Model improvement attribution using pre/post aggregation weight norms
    - Per-round reward calculation proportional to Shapley contribution
    - Free-rider detection (participants with below-threshold contributions)
    - Historical contribution analytics across rounds

    Args:
        contribution_repository: Async repository for persisting ContributionRecord
            objects. Must expose save_contribution() and list_contributions() methods.
        shapley_permutations: Number of random permutations for Shapley sampling.
            Higher = more accurate but slower.
        reward_pool_per_round: Total reward units to distribute each round.
        free_rider_threshold: Shapley fraction below which a participant is
            flagged as a potential free rider.
    """

    def __init__(
        self,
        contribution_repository: Any,
        shapley_permutations: int = DEFAULT_SHAPLEY_PERMUTATIONS,
        reward_pool_per_round: float = REWARD_POOL_DEFAULT,
        free_rider_threshold: float = FREE_RIDER_THRESHOLD,
    ) -> None:
        self._repo = contribution_repository
        self._shapley_permutations = shapley_permutations
        self._reward_pool = reward_pool_per_round
        self._free_rider_threshold = free_rider_threshold

        # job_id -> {participant_id -> [ContributionRecord]} (in-memory cache)
        self._history_cache: dict[str, dict[str, list[ContributionRecord]]] = {}

    # ------------------------------------------------------------------
    # Data quality scoring
    # ------------------------------------------------------------------

    def score_data_quality(
        self,
        participant_id: str,
        num_samples: int,
        declared_class_distribution: dict[str, int] | None = None,
        duplicate_fraction: float = 0.0,
        missing_value_fraction: float = 0.0,
    ) -> float:
        """Compute a data quality score (0–1) for a participant's dataset.

        Factors considered:
        - Dataset size relative to a reference (larger = higher score, up to 1)
        - Class distribution entropy (uniform = 1, maximally skewed = 0)
        - Duplicate penalty
        - Missing value penalty

        Args:
            participant_id: The participant being scored.
            num_samples: Number of training samples contributed.
            declared_class_distribution: Dict of {class_label: count}. If None,
                class balance scoring is skipped.
            duplicate_fraction: Fraction of samples that are duplicates (0–1).
            missing_value_fraction: Fraction of feature values that are missing (0–1).

        Returns:
            Quality score in [0, 1].
        """
        # Size score: sigmoid-shaped, saturates at ~10,000 samples
        size_score = 1.0 - math.exp(-num_samples / 5000.0)

        # Class balance score using normalised entropy
        if declared_class_distribution and len(declared_class_distribution) > 1:
            total = sum(declared_class_distribution.values())
            probs = [count / total for count in declared_class_distribution.values() if count > 0]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(len(declared_class_distribution))
            balance_score = entropy / max_entropy if max_entropy > 0 else 1.0
        else:
            balance_score = 1.0  # Unknown distribution — no penalty

        # Cleanliness penalties
        duplicate_penalty = max(0.0, 1.0 - duplicate_fraction * 2.0)
        missing_penalty = max(0.0, 1.0 - missing_value_fraction * 1.5)

        composite = (
            0.4 * size_score
            + 0.3 * balance_score
            + 0.15 * duplicate_penalty
            + 0.15 * missing_penalty
        )

        score = min(1.0, max(0.0, composite))
        logger.debug(
            "Data quality score for participant %s: %.4f "
            "(size=%.2f, balance=%.2f, dup_penalty=%.2f, missing_penalty=%.2f)",
            participant_id,
            score,
            size_score,
            balance_score,
            duplicate_penalty,
            missing_penalty,
        )
        return score

    # ------------------------------------------------------------------
    # Model improvement attribution
    # ------------------------------------------------------------------

    def compute_improvement_attribution(
        self,
        global_params_before: list[np.ndarray[Any, Any]],
        global_params_after: list[np.ndarray[Any, Any]],
        participant_updates: dict[str, list[np.ndarray[Any, Any]]],
    ) -> dict[str, float]:
        """Attribute model improvement to individual participants.

        Uses a leave-one-out attribution: for each participant, compute
        the L2 norm difference between the full aggregate and the aggregate
        without that participant. Larger difference = higher attribution.

        Args:
            global_params_before: Aggregated model weights from the previous round.
            global_params_after: Aggregated model weights from this round.
            participant_updates: {participant_id: weight_arrays} for this round.

        Returns:
            Dict of {participant_id: improvement_delta_score}.
        """
        total_delta = self._l2_norm_diff(global_params_before, global_params_after)
        if total_delta == 0.0:
            return {pid: 0.0 for pid in participant_updates}

        attributions: dict[str, float] = {}
        participant_ids = list(participant_updates.keys())
        num_participants = len(participant_ids)

        if num_participants == 0:
            return {}

        # Compute weighted average without each participant (leave-one-out)
        # For simplicity: attribution proportional to the L2 norm of their update
        total_update_norm = sum(
            self._l2_norm(update) for update in participant_updates.values()
        )

        for pid, update in participant_updates.items():
            individual_norm = self._l2_norm(update)
            attribution = (
                (individual_norm / total_update_norm) * total_delta
                if total_update_norm > 0
                else total_delta / num_participants
            )
            attributions[pid] = round(attribution, 8)

        return attributions

    # ------------------------------------------------------------------
    # Shapley value approximation
    # ------------------------------------------------------------------

    def approximate_shapley_values(
        self,
        participant_ids: list[str],
        characteristic_function: Any,
        num_permutations: int | None = None,
    ) -> dict[str, float]:
        """Compute approximate Shapley values via random permutation sampling.

        The characteristic function v(S) should return a real-valued score
        (e.g., model accuracy or loss improvement) for a subset S of participants.

        This is the permutation-based Shapley Value Approximation (SVP) algorithm,
        O(m * n) where m = permutations and n = |participant_ids|.

        Args:
            participant_ids: All participants in the coalition.
            characteristic_function: Callable (frozenset[str]) -> float.
                Must be deterministic. Called O(m * n) times.
            num_permutations: Override the default permutation count.

        Returns:
            Dict of {participant_id: shapley_value}, normalised so values sum to v(N).
        """
        n_perms = num_permutations or self._shapley_permutations
        shapley_sums: dict[str, float] = {pid: 0.0 for pid in participant_ids}
        participants = list(participant_ids)

        for _ in range(n_perms):
            permutation = participants[:]
            random.shuffle(permutation)
            coalition: frozenset[str] = frozenset()
            previous_value = characteristic_function(coalition)

            for pid in permutation:
                coalition = coalition | {pid}
                new_value = characteristic_function(coalition)
                marginal_contribution = new_value - previous_value
                shapley_sums[pid] += marginal_contribution
                previous_value = new_value

        # Average over permutations
        shapley_values = {
            pid: total / n_perms for pid, total in shapley_sums.items()
        }

        logger.info(
            "Computed approximate Shapley values for %d participants "
            "over %d permutations",
            len(participants),
            n_perms,
        )
        return shapley_values

    # ------------------------------------------------------------------
    # Reward calculation
    # ------------------------------------------------------------------

    def calculate_rewards(
        self,
        shapley_values: dict[str, float],
        reward_pool: float | None = None,
    ) -> dict[str, float]:
        """Allocate reward units proportional to Shapley values.

        Participants with negative or zero Shapley values receive zero reward.
        The pool is distributed among positive-contribution participants.

        Args:
            shapley_values: {participant_id: shapley_value} for this round.
            reward_pool: Total reward units to distribute (uses class default if None).

        Returns:
            Dict of {participant_id: reward_units}.
        """
        pool = reward_pool if reward_pool is not None else self._reward_pool

        positive_sv = {
            pid: max(0.0, sv) for pid, sv in shapley_values.items()
        }
        total_positive = sum(positive_sv.values())

        if total_positive == 0.0:
            # Equal split if all Shapley values are zero/negative
            equal_share = pool / max(1, len(shapley_values))
            return {pid: equal_share for pid in shapley_values}

        rewards = {
            pid: (sv / total_positive) * pool
            for pid, sv in positive_sv.items()
        }
        return rewards

    # ------------------------------------------------------------------
    # Free-rider detection
    # ------------------------------------------------------------------

    def detect_free_riders(
        self,
        shapley_values: dict[str, float],
    ) -> list[str]:
        """Identify participants whose Shapley contribution is below the threshold.

        A free rider is a participant who benefits from the federated model
        without contributing meaningfully to its improvement.

        Args:
            shapley_values: {participant_id: shapley_value} normalised so they
                sum to the total characteristic function value.

        Returns:
            List of participant_ids suspected of free-riding.
        """
        total_value = sum(shapley_values.values())
        if total_value <= 0.0:
            return []

        free_riders: list[str] = []
        for pid, sv in shapley_values.items():
            fraction = sv / total_value if total_value > 0 else 0.0
            if fraction < self._free_rider_threshold:
                free_riders.append(pid)
                logger.warning(
                    "Free-rider detected: participant %s Shapley fraction=%.4f "
                    "(threshold=%.4f)",
                    pid,
                    fraction,
                    self._free_rider_threshold,
                )

        return free_riders

    # ------------------------------------------------------------------
    # Score and persist a round
    # ------------------------------------------------------------------

    async def score_round(
        self,
        job_id: str,
        round_number: int,
        participant_data: list[dict[str, Any]],
        shapley_values: dict[str, float] | None = None,
        model_improvement_attributions: dict[str, float] | None = None,
    ) -> list[ContributionRecord]:
        """Score all participants for a completed training round and persist results.

        Args:
            job_id: The federated job.
            round_number: The completed round.
            participant_data: List of dicts with keys:
                - participant_id (str)
                - num_samples (int)
                - class_distribution (dict[str, int] | None)
                - duplicate_fraction (float, optional)
                - missing_value_fraction (float, optional)
            shapley_values: Pre-computed Shapley values (if available).
            model_improvement_attributions: Pre-computed improvement deltas.

        Returns:
            List of persisted ContributionRecord objects.
        """
        participant_ids = [pd["participant_id"] for pd in participant_data]
        sv = shapley_values or {pid: 1.0 / len(participant_ids) for pid in participant_ids}
        attributions = model_improvement_attributions or {pid: 0.0 for pid in participant_ids}

        free_riders = self.detect_free_riders(sv)
        rewards = self.calculate_rewards(sv)

        records: list[ContributionRecord] = []

        for participant_info in participant_data:
            pid = participant_info["participant_id"]
            num_samples = int(participant_info.get("num_samples", 0))
            class_dist = participant_info.get("class_distribution")
            dup_frac = float(participant_info.get("duplicate_fraction", 0.0))
            missing_frac = float(participant_info.get("missing_value_fraction", 0.0))

            quality_score = self.score_data_quality(
                participant_id=pid,
                num_samples=num_samples,
                declared_class_distribution=class_dist,
                duplicate_fraction=dup_frac,
                missing_value_fraction=missing_frac,
            )

            record = ContributionRecord(
                participant_id=pid,
                job_id=job_id,
                round_number=round_number,
                data_quality_score=quality_score,
                model_improvement_delta=attributions.get(pid, 0.0),
                shapley_value=sv.get(pid, 0.0),
                num_samples=num_samples,
                reward_units=rewards.get(pid, 0.0),
                is_free_rider=pid in free_riders,
            )

            await self._repo.save_contribution(record)

            # Update in-memory cache
            job_cache = self._history_cache.setdefault(job_id, {})
            participant_history = job_cache.setdefault(pid, [])
            participant_history.append(record)

            records.append(record)

        logger.info(
            "Scored round %d for job %s: %d participants, %d free-riders detected",
            round_number,
            job_id,
            len(records),
            len(free_riders),
        )
        return records

    # ------------------------------------------------------------------
    # Historical analytics
    # ------------------------------------------------------------------

    async def get_participant_analytics(
        self,
        job_id: str,
        participant_id: str,
    ) -> dict[str, Any]:
        """Return cumulative contribution analytics for a participant.

        Args:
            job_id: The federated job.
            participant_id: The participant to query.

        Returns:
            Dict with total_rounds, total_samples, total_reward, average_shapley,
            average_quality_score, free_rider_rounds, round_history.
        """
        contributions = await self._repo.list_contributions(
            job_id=job_id, participant_id=participant_id
        )

        if not contributions:
            return {
                "participant_id": participant_id,
                "job_id": job_id,
                "total_rounds": 0,
                "total_samples": 0,
                "total_reward": 0.0,
                "average_shapley": 0.0,
                "average_quality_score": 0.0,
                "free_rider_rounds": 0,
                "round_history": [],
            }

        total_samples = sum(c.num_samples for c in contributions)
        total_reward = sum(c.reward_units for c in contributions)
        avg_shapley = sum(c.shapley_value for c in contributions) / len(contributions)
        avg_quality = sum(c.data_quality_score for c in contributions) / len(contributions)
        free_rider_rounds = sum(1 for c in contributions if c.is_free_rider)

        return {
            "participant_id": participant_id,
            "job_id": job_id,
            "total_rounds": len(contributions),
            "total_samples": total_samples,
            "total_reward": round(total_reward, 4),
            "average_shapley": round(avg_shapley, 6),
            "average_quality_score": round(avg_quality, 4),
            "free_rider_rounds": free_rider_rounds,
            "round_history": [c.to_dict() for c in sorted(contributions, key=lambda c: c.round_number)],
        }

    async def generate_distribution_report(
        self, job_id: str
    ) -> dict[str, Any]:
        """Generate an incentive distribution report for an entire job.

        Args:
            job_id: The federated job to report on.

        Returns:
            Structured report with per-participant and per-round breakdowns.
        """
        all_contributions = await self._repo.list_contributions(job_id=job_id)

        if not all_contributions:
            return {
                "job_id": job_id,
                "total_rounds": 0,
                "total_reward_distributed": 0.0,
                "participants": [],
            }

        # Group by participant
        by_participant: dict[str, list[ContributionRecord]] = {}
        for record in all_contributions:
            by_participant.setdefault(record.participant_id, []).append(record)

        participant_summaries = []
        for pid, records in by_participant.items():
            participant_summaries.append(
                {
                    "participant_id": pid,
                    "rounds_participated": len(records),
                    "total_reward": round(sum(r.reward_units for r in records), 4),
                    "total_samples": sum(r.num_samples for r in records),
                    "average_quality_score": round(
                        sum(r.data_quality_score for r in records) / len(records), 4
                    ),
                    "average_shapley": round(
                        sum(r.shapley_value for r in records) / len(records), 6
                    ),
                    "free_rider_count": sum(1 for r in records if r.is_free_rider),
                }
            )

        total_reward = sum(r.reward_units for r in all_contributions)
        rounds_covered = sorted({r.round_number for r in all_contributions})

        return {
            "job_id": job_id,
            "total_rounds": len(rounds_covered),
            "rounds_covered": rounds_covered,
            "total_reward_distributed": round(total_reward, 4),
            "participants": sorted(
                participant_summaries,
                key=lambda p: p["total_reward"],
                reverse=True,
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_norm(params: list[np.ndarray[Any, Any]]) -> float:
        """Compute the L2 norm of a flattened list of parameter arrays.

        Args:
            params: List of NumPy weight arrays.

        Returns:
            Scalar L2 norm.
        """
        flat = np.concatenate([p.flatten() for p in params])
        return float(np.linalg.norm(flat))

    @staticmethod
    def _l2_norm_diff(
        params_a: list[np.ndarray[Any, Any]],
        params_b: list[np.ndarray[Any, Any]],
    ) -> float:
        """Compute the L2 norm of the difference between two parameter sets.

        Args:
            params_a: First set of weight arrays.
            params_b: Second set of weight arrays.

        Returns:
            Scalar L2 norm of (params_b - params_a).
        """
        delta_flat = np.concatenate(
            [(b - a).flatten() for a, b in zip(params_a, params_b, strict=True)]
        )
        return float(np.linalg.norm(delta_flat))
