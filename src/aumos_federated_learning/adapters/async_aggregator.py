"""Asynchronous federated aggregation implementations.

Gap #149: Missing Asynchronous FL Support.

Provides FedAsync (Xie et al. 2019, arXiv:1903.03934) and plain ASGD for
asynchronous model update aggregation without waiting for straggler participants.

Reference:
    Xie, C., Koyejo, O., & Gupta, I. (2019). Asynchronous Federated Optimization.
    arXiv:1903.03934.
"""

from __future__ import annotations

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class FedAsyncAggregator:
    """Asynchronous federated averaging with staleness-aware weighting.

    Implements FedAsync: each client update is weighted by a staleness factor
    that decays with the number of rounds the update lags behind the global model::

        α(τ) = 1 / (1 + τ) ^ staleness_factor

    Updates are buffered until ``buffer_size`` updates accumulate, then
    a weighted average produces the new global model.

    Args:
        staleness_factor: Controls penalty steepness for stale updates.
            ``staleness_factor=1.0`` gives linear decay; higher values penalise
            stale updates more aggressively.
        buffer_size: Number of updates to accumulate before aggregating.
            ``buffer_size=1`` aggregates after every received update (fully async).
    """

    def __init__(
        self,
        staleness_factor: float = 1.0,
        buffer_size: int = 1,
    ) -> None:
        self.staleness_factor = staleness_factor
        self.buffer_size = buffer_size
        # Each entry: (flat_weights, num_examples, staleness_weight)
        self._update_buffer: list[tuple[np.ndarray[int, np.dtype[np.float32]], int, float]] = []
        self._global_weights: np.ndarray[int, np.dtype[np.float32]] | None = None
        self._current_round: int = 0

    @property
    def current_round(self) -> int:
        """Return the current global round number."""
        return self._current_round

    def add_update(
        self,
        local_weights: np.ndarray[int, np.dtype[np.float32]],
        num_examples: int,
        client_round: int,
    ) -> np.ndarray[int, np.dtype[np.float32]] | None:
        """Add a participant update to the buffer.

        Triggers aggregation if the buffer has reached ``buffer_size``.

        Args:
            local_weights: Flat NumPy array of client model weights.
            num_examples: Number of training examples used by this client.
            client_round: The global round number when the client started training.
                Used to compute staleness = current_round - client_round.

        Returns:
            Updated global weights if aggregation was triggered, else ``None``.
        """
        staleness = max(0, self._current_round - client_round)
        staleness_weight = 1.0 / (1.0 + staleness) ** self.staleness_factor
        self._update_buffer.append((local_weights, num_examples, staleness_weight))

        logger.debug(
            "Update buffered",
            staleness=staleness,
            staleness_weight=staleness_weight,
            buffer_len=len(self._update_buffer),
        )

        if len(self._update_buffer) >= self.buffer_size:
            return self._aggregate_buffer()
        return None

    def _aggregate_buffer(self) -> np.ndarray[int, np.dtype[np.float32]]:
        """Perform staleness-weighted average of buffered updates.

        Returns:
            New global weights as a flat NumPy array.
        """
        total_weight = sum(
            n_examples * sw for _, n_examples, sw in self._update_buffer
        )
        if total_weight == 0.0:
            total_weight = 1.0  # Guard against divide-by-zero

        aggregated: np.ndarray[int, np.dtype[np.float32]] = sum(  # type: ignore[assignment]
            weights * (n_examples * sw / total_weight)
            for weights, n_examples, sw in self._update_buffer
        )
        self._update_buffer.clear()
        self._current_round += 1
        self._global_weights = aggregated

        logger.info(
            "Async aggregation completed",
            new_round=self._current_round,
        )
        return aggregated

    def get_global_weights(self) -> np.ndarray[int, np.dtype[np.float32]] | None:
        """Return the current global model weights.

        Returns:
            Global weights, or ``None`` if no aggregation has occurred yet.
        """
        return self._global_weights

    def reset(self) -> None:
        """Reset aggregator state (e.g. for a new job)."""
        self._update_buffer.clear()
        self._global_weights = None
        self._current_round = 0


class ASGDAggregator:
    """Asynchronous SGD aggregator.

    Applies each received gradient update immediately to the global model
    without any buffering or staleness weighting. Suitable for single-server
    scenarios with homogeneous participants.

    Args:
        learning_rate: Global model learning rate for each update application.
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self._global_weights: np.ndarray[int, np.dtype[np.float32]] | None = None
        self._updates_applied: int = 0

    def apply_update(
        self,
        gradient: np.ndarray[int, np.dtype[np.float32]],
        num_examples: int,
    ) -> np.ndarray[int, np.dtype[np.float32]]:
        """Apply a gradient update to the global model immediately.

        Args:
            gradient: Weight delta (not the full weights — the gradient) from a client.
            num_examples: Number of examples used by this client (for logging only).

        Returns:
            Updated global weights.
        """
        if self._global_weights is None:
            self._global_weights = np.zeros_like(gradient)

        self._global_weights = self._global_weights - self.learning_rate * gradient
        self._updates_applied += 1

        logger.debug(
            "ASGD update applied",
            updates_applied=self._updates_applied,
            num_examples=num_examples,
        )
        return self._global_weights

    def get_global_weights(self) -> np.ndarray[int, np.dtype[np.float32]] | None:
        """Return the current global model weights."""
        return self._global_weights

    def reset(self) -> None:
        """Reset aggregator state."""
        self._global_weights = None
        self._updates_applied = 0
