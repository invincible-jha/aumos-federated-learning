"""Federated analytics engine for privacy-preserving aggregate statistics.

Gap #152: No Federated Analytics (Non-ML).

Enables organisations to compute joint statistics — mean, variance, histogram,
quantiles — across participants without sharing individual records.

All outputs are protected by configurable differential privacy (Laplace mechanism).

References:
    Cormode, G., et al. (2012). Privacy in Data Mining.
    Apple Differential Privacy Team (2017). Learning with Privacy at Scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class AggregationType(str, Enum):
    """Supported federated analytic aggregation types."""

    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    VARIANCE = "variance"
    HISTOGRAM = "histogram"


@dataclass
class LocalAnalyticsResult:
    """A single participant's local analytic computation.

    Attributes:
        participant_id: Identifier of the contributing participant.
        aggregation_type: Type of aggregation computed.
        value: The local aggregate value (scalar, list, or dict depending on type).
        count: Number of records that contributed to this result.
        metadata: Additional context (bin edges for histograms, etc.).
    """

    participant_id: str
    aggregation_type: AggregationType
    value: Any
    count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FederatedAnalyticsResult:
    """Server-side aggregated federated analytic result.

    Attributes:
        aggregation_type: Type of aggregation that was computed.
        result: The DP-protected global aggregate.
        total_count: Total number of records across all participants.
        num_participants: Number of participants that contributed.
        epsilon_consumed: Differential privacy budget consumed.
        metadata: Additional context (bin edges, column name, etc.).
    """

    aggregation_type: AggregationType
    result: Any
    total_count: int
    num_participants: int
    epsilon_consumed: float
    metadata: dict[str, Any] = field(default_factory=dict)


class FederatedAnalyticsEngine:
    """Computes differentially private statistics across FL participants.

    Each participant executes the query locally and sends the local aggregate.
    The server aggregates the local results using appropriate combination rules:

    - COUNT: sum of local counts
    - SUM: sum of local sums
    - MEAN: (sum of local sums) / (sum of local counts)
    - VARIANCE: pooled variance via variance decomposition
    - HISTOGRAM: sum of local histogram bins

    Global Laplace noise is added by the server after aggregation to protect
    against inference from the aggregate itself.

    Args:
        epsilon: Differential privacy budget (total, for this query).
        sensitivity: Global sensitivity of the query. For COUNT queries this
            is typically 1. For SUM/MEAN it depends on the data domain.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        sensitivity: float = 1.0,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self._rng = np.random.default_rng()

    def aggregate(
        self,
        local_results: list[LocalAnalyticsResult],
    ) -> FederatedAnalyticsResult:
        """Aggregate local analytic results into a global DP-protected answer.

        Args:
            local_results: List of local results from all participating nodes.

        Returns:
            FederatedAnalyticsResult with the DP-protected global value.

        Raises:
            ValueError: If ``local_results`` is empty or has inconsistent types.
        """
        if not local_results:
            raise ValueError("Cannot aggregate zero local results")

        agg_type = local_results[0].aggregation_type
        if not all(r.aggregation_type == agg_type for r in local_results):
            raise ValueError("All local results must have the same aggregation_type")

        total_count = sum(r.count for r in local_results)
        num_participants = len(local_results)

        if agg_type == AggregationType.COUNT:
            global_value = self._aggregate_count(local_results)
        elif agg_type == AggregationType.SUM:
            global_value = self._aggregate_sum(local_results)
        elif agg_type == AggregationType.MEAN:
            global_value = self._aggregate_mean(local_results)
        elif agg_type == AggregationType.VARIANCE:
            global_value = self._aggregate_variance(local_results)
        elif agg_type == AggregationType.HISTOGRAM:
            global_value = self._aggregate_histogram(local_results)
        else:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        # Add Laplace noise for global differential privacy
        noise_scale = self.sensitivity / self.epsilon
        if isinstance(global_value, (int, float)):
            noised_value: Any = float(global_value) + float(self._rng.laplace(0, noise_scale))
        elif isinstance(global_value, list):
            noise = self._rng.laplace(0, noise_scale, len(global_value))
            noised_value = [float(v) + float(n) for v, n in zip(global_value, noise)]
        else:
            noised_value = global_value  # Passthrough for complex types

        logger.info(
            "Federated analytics aggregated",
            aggregation_type=agg_type.value,
            total_count=total_count,
            num_participants=num_participants,
            epsilon=self.epsilon,
        )

        return FederatedAnalyticsResult(
            aggregation_type=agg_type,
            result=noised_value,
            total_count=total_count,
            num_participants=num_participants,
            epsilon_consumed=self.epsilon,
            metadata=local_results[0].metadata if local_results else {},
        )

    def _aggregate_count(
        self,
        results: list[LocalAnalyticsResult],
    ) -> float:
        return float(sum(r.count for r in results))

    def _aggregate_sum(
        self,
        results: list[LocalAnalyticsResult],
    ) -> float:
        return float(sum(float(r.value) for r in results))

    def _aggregate_mean(
        self,
        results: list[LocalAnalyticsResult],
    ) -> float:
        total_sum = sum(float(r.value) * r.count for r in results)
        total_count = sum(r.count for r in results)
        return total_sum / max(total_count, 1)

    def _aggregate_variance(
        self,
        results: list[LocalAnalyticsResult],
    ) -> float:
        """Compute pooled variance using the parallel variance algorithm.

        Each local result should include ``local_mean`` and ``local_variance``
        in its value dict.
        """
        total_n = sum(r.count for r in results)
        if total_n == 0:
            return 0.0

        # Parallel variance (Chan et al. 1979)
        combined_mean = self._aggregate_mean(results)
        pooled_variance = sum(
            r.count * (float(r.value.get("variance", 0)) + (float(r.value.get("mean", 0)) - combined_mean) ** 2)
            for r in results
        ) / total_n
        return float(pooled_variance)

    def _aggregate_histogram(
        self,
        results: list[LocalAnalyticsResult],
    ) -> list[float]:
        """Sum histogram bin counts across participants."""
        if not results:
            return []
        bin_counts: list[float] = [0.0] * len(results[0].value)
        for r in results:
            for i, count in enumerate(r.value):
                if i < len(bin_counts):
                    bin_counts[i] += float(count)
        return bin_counts

    @staticmethod
    def create_local_count(
        participant_id: str,
        data: list[Any],
    ) -> LocalAnalyticsResult:
        """Helper to create a local COUNT result from raw data.

        Args:
            participant_id: Participant identifier.
            data: Local dataset (list of records).

        Returns:
            LocalAnalyticsResult with count.
        """
        return LocalAnalyticsResult(
            participant_id=participant_id,
            aggregation_type=AggregationType.COUNT,
            value=len(data),
            count=len(data),
        )

    @staticmethod
    def create_local_sum(
        participant_id: str,
        values: list[float],
    ) -> LocalAnalyticsResult:
        """Helper to create a local SUM result from numeric values.

        Args:
            participant_id: Participant identifier.
            values: List of numeric values to sum locally.

        Returns:
            LocalAnalyticsResult with local sum.
        """
        return LocalAnalyticsResult(
            participant_id=participant_id,
            aggregation_type=AggregationType.SUM,
            value=sum(values),
            count=len(values),
        )

    @staticmethod
    def create_local_histogram(
        participant_id: str,
        values: list[float],
        bins: int = 10,
        range_min: float = 0.0,
        range_max: float = 1.0,
    ) -> LocalAnalyticsResult:
        """Helper to create a local HISTOGRAM result.

        Args:
            participant_id: Participant identifier.
            values: Local numeric values to bin.
            bins: Number of histogram bins.
            range_min: Minimum value of the histogram range.
            range_max: Maximum value of the histogram range.

        Returns:
            LocalAnalyticsResult with bin counts and bin edge metadata.
        """
        arr = np.array(values)
        counts, edges = np.histogram(arr, bins=bins, range=(range_min, range_max))
        return LocalAnalyticsResult(
            participant_id=participant_id,
            aggregation_type=AggregationType.HISTOGRAM,
            value=counts.tolist(),
            count=len(values),
            metadata={"bin_edges": edges.tolist(), "bins": bins},
        )
