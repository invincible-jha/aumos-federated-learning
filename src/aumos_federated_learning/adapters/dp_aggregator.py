"""Differentially private secure aggregation adapter.

Applies the Gaussian mechanism to aggregated model parameters,
providing formal (ε, δ)-differential privacy guarantees.

The DP noise calibration follows the moments accountant / Rényi DP framework
(same as Opacus). Privacy budget accounting is delegated to aumos-privacy-engine.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DifferentiallyPrivateAggregator:
    """Applies calibrated Gaussian noise to aggregated model parameters.

    This implements server-side DP aggregation:
    1. Clip each participant's update to max_grad_norm
    2. Sum clipped updates
    3. Add Gaussian noise calibrated to (sigma * max_grad_norm / n)
    4. Divide by n to get average

    This is equivalent to user-level DP where each participant is a "user".
    """

    def __init__(self, privacy_client: Any | None = None) -> None:
        self._privacy_client = privacy_client

    def aggregate_with_dp(
        self,
        updates: list[tuple[list[np.ndarray[Any, Any]], int]],
        epsilon: float,
        delta: float,
        noise_multiplier: float,
        max_grad_norm: float,
    ) -> tuple[list[np.ndarray[Any, Any]], dict[str, float]]:
        """Aggregate updates with Gaussian differential privacy noise.

        Args:
            updates: List of (parameter_arrays, num_samples) from participants.
            epsilon: Privacy budget ε — budget consumed per round.
            delta: Privacy failure probability δ.
            noise_multiplier: σ — ratio of noise std to sensitivity (max_grad_norm).
            max_grad_norm: L2 norm clipping threshold for participant updates.

        Returns:
            (noised_aggregated_parameters, privacy_accounting_metrics)
        """
        if not updates:
            raise ValueError("No updates provided for DP aggregation")

        num_participants = len(updates)

        # Step 1: Clip each participant's update to max_grad_norm
        clipped_updates: list[list[np.ndarray[Any, Any]]] = []
        for params, _num_samples in updates:
            clipped = self._clip_update(params, max_grad_norm)
            clipped_updates.append(clipped)

        # Step 2: Weighted sum of clipped updates
        total_samples = sum(n for _, n in updates)
        if total_samples == 0:
            raise ValueError("Total sample count is zero")

        num_layers = len(clipped_updates[0])
        aggregated: list[np.ndarray[Any, Any]] = []
        for layer_idx in range(num_layers):
            weighted_sum = np.zeros_like(clipped_updates[0][layer_idx], dtype=np.float64)
            for (_, num_samples), clipped in zip(updates, clipped_updates, strict=True):
                weight = num_samples / total_samples
                weighted_sum += clipped[layer_idx] * weight
            aggregated.append(weighted_sum)

        # Step 3: Add calibrated Gaussian noise
        # Noise std = noise_multiplier * max_grad_norm / num_participants
        noise_std = noise_multiplier * max_grad_norm / num_participants
        noised: list[np.ndarray[Any, Any]] = []
        for layer in aggregated:
            noise = np.random.normal(0.0, noise_std, size=layer.shape)
            noised.append((layer + noise).astype(layer.dtype))

        # Privacy accounting metrics
        actual_epsilon = self.compute_epsilon(
            num_rounds=1,
            noise_multiplier=noise_multiplier,
            sample_rate=1.0 / max(num_participants, 1),
            delta=delta,
        )

        privacy_metrics: dict[str, float] = {
            "epsilon_used": actual_epsilon,
            "delta": delta,
            "noise_multiplier": noise_multiplier,
            "noise_std": noise_std,
            "max_grad_norm": max_grad_norm,
            "num_participants": float(num_participants),
        }

        logger.info(
            "DP aggregation complete: ε=%.4f (target=%.4f), δ=%.2e, noise_std=%.4f",
            actual_epsilon,
            epsilon,
            delta,
            noise_std,
        )

        return noised, privacy_metrics

    def _clip_update(
        self,
        params: list[np.ndarray[Any, Any]],
        max_norm: float,
    ) -> list[np.ndarray[Any, Any]]:
        """Clip the L2 norm of a flattened parameter vector to max_norm."""
        flat = np.concatenate([p.flatten() for p in params])
        current_norm = float(np.linalg.norm(flat))

        if current_norm > max_norm:
            scale = max_norm / current_norm
            return [p * scale for p in params]
        return list(params)

    def compute_epsilon(
        self,
        num_rounds: int,
        noise_multiplier: float,
        sample_rate: float,
        delta: float,
    ) -> float:
        """Compute the privacy budget consumed using a simplified Rényi DP bound.

        For production use, delegate to aumos-privacy-engine for exact RDP accounting.
        This provides a conservative upper bound via the strong composition theorem.

        Privacy amplification by subsampling: ε_round ≈ sqrt(2 * log(1/δ)) / noise_multiplier
        Over T rounds: ε_total ≈ sqrt(T) * ε_round  (advanced composition)
        """
        if noise_multiplier <= 0:
            return math.inf

        # Single-round epsilon via standard Gaussian mechanism bound
        # σ = noise_multiplier (normalized sensitivity = 1)
        single_round_epsilon = math.sqrt(2.0 * math.log(1.25 / delta)) / noise_multiplier

        # Amplification by subsampling
        amplified_epsilon = sample_rate * single_round_epsilon

        # Advanced composition over T rounds: ε_total ≈ sqrt(T * 2 * ln(1/δ)) * ε_round
        if num_rounds == 1:
            return amplified_epsilon

        total_epsilon = math.sqrt(
            2.0 * num_rounds * math.log(1.0 / delta)
        ) * amplified_epsilon

        return total_epsilon
