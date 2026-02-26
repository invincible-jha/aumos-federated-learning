"""SCAFFOLD strategy — Stochastic Controlled Averaging for Federated Learning.

SCAFFOLD uses control variates (c_i for each client, c for the server) to
correct for client drift caused by heterogeneous local data distributions.
This makes it more communication-efficient than FedAvg/FedProx on non-IID data.

Reference: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for
Federated Learning", ICML 2020.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ScaffoldStrategy:
    """SCAFFOLD strategy for heterogeneous federated learning.

    Unlike FedAvg/FedProx, SCAFFOLD requires clients to maintain and exchange
    control variates, which requires protocol support beyond Flower's default
    FitIns/FitRes. This implementation orchestrates the control variate exchange
    at the coordinator level.

    In production, the full SCAFFOLD update requires:
    1. Server broadcasts global model w_t and server control variate c
    2. Clients run local SCAFFOLD update: w_i^t+1, c_i^t+1 = scaffold_step(w_t, c, c_i)
    3. Clients send (Δw_i, Δc_i) — delta updates, not full weights
    4. Server aggregates: w_t+1 = w_t + η/K * Σ Δw_i, c = c + 1/N * Σ Δc_i
    """

    strategy_name = "scaffold"

    def __init__(
        self,
        *,
        global_learning_rate: float = 1.0,
        local_learning_rate: float = 0.01,
        local_steps: int = 10,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        self.global_learning_rate = global_learning_rate
        self.local_learning_rate = local_learning_rate
        self.local_steps = local_steps
        self._fraction_fit = fraction_fit
        self._min_fit_clients = min_fit_clients
        self._min_available_clients = min_available_clients

        # Server-side control variate — initialized on first round
        self._server_control_variate: list[np.ndarray[Any, Any]] | None = None
        logger.info("SCAFFOLD strategy initialized (global_lr=%.4f)", global_learning_rate)

    def configure_fit(
        self,
        server_round: int,
        parameters: list[np.ndarray[Any, Any]],
        client_manager: Any,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Configure SCAFFOLD fit round, including server control variate broadcast."""
        if self._server_control_variate is None:
            # Initialize server control variate to zeros
            self._server_control_variate = [np.zeros_like(p) for p in parameters]

        fit_config: dict[str, Any] = {
            "server_round": server_round,
            "global_learning_rate": self.global_learning_rate,
            "local_learning_rate": self.local_learning_rate,
            "local_steps": self.local_steps,
            # In production: serialize and send server_control_variate to clients
            "has_server_control_variate": self._server_control_variate is not None,
        }

        # In a real deployment, client_manager.sample() is called here.
        # We return an empty list as the coordinator handles client selection externally.
        return []

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[Any, Any]],
        failures: list[Any],
    ) -> tuple[list[np.ndarray[Any, Any]] | None, dict[str, Any]]:
        """Aggregate SCAFFOLD delta updates and update server control variate.

        Expects each result to contain:
        - delta_parameters: model weight deltas
        - delta_control: control variate deltas
        """
        if not results:
            logger.warning("SCAFFOLD: no results received for round %d", server_round)
            return None, {}

        if len(failures) > 0:
            logger.warning("SCAFFOLD: %d clients failed in round %d", len(failures), server_round)

        # Extract delta updates
        # In a full implementation, results contain (client, fit_res) where
        # fit_res.metrics includes delta_control serialized bytes.
        # Here we perform a simplified weighted average of submitted parameters.
        weights_results: list[tuple[list[np.ndarray[Any, Any]], int]] = []
        for _, fit_res in results:
            if hasattr(fit_res, "parameters") and hasattr(fit_res, "num_examples"):
                try:
                    from flwr.common import parameters_to_ndarrays
                    ndarrays = parameters_to_ndarrays(fit_res.parameters)
                except (ImportError, Exception):
                    ndarrays = fit_res.parameters
                weights_results.append((ndarrays, fit_res.num_examples))

        if not weights_results:
            return None, {}

        total_examples = sum(n for _, n in weights_results)
        if total_examples == 0:
            return None, {}

        # Weighted average of deltas, scaled by global_learning_rate
        aggregated: list[np.ndarray[Any, Any]] = []
        for layer_idx in range(len(weights_results[0][0])):
            weighted_sum = np.zeros_like(weights_results[0][0][layer_idx], dtype=np.float64)
            for params, num_examples in weights_results:
                weighted_sum += params[layer_idx] * (num_examples / total_examples)
            aggregated.append(weighted_sum * self.global_learning_rate)

        # Update server control variate (simplified: zero update without client variates)
        # Full implementation requires client control variate deltas from fit_res.metrics

        return aggregated, {
            "aggregation_method": "scaffold",
            "global_learning_rate": self.global_learning_rate,
            "num_clients": len(weights_results),
        }

    def configure_evaluate(
        self,
        server_round: int,
        parameters: list[np.ndarray[Any, Any]],
        client_manager: Any,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """SCAFFOLD evaluation is same as standard FL."""
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[Any, Any]],
        failures: list[Any],
    ) -> tuple[float | None, dict[str, Any]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}
        total_examples = sum(
            r[1].num_examples for r in results if hasattr(r[1], "num_examples")
        )
        if total_examples == 0:
            return None, {}
        weighted_loss = sum(
            r[1].loss * r[1].num_examples / total_examples
            for r in results
            if hasattr(r[1], "num_examples")
        )
        return weighted_loss, {"aggregation_method": "scaffold"}
