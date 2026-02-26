"""Federated Averaging (FedAvg) strategy via Flower.

FedAvg is the canonical FL algorithm: the server averages client model weights
weighted by each client's number of training samples.

Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data", AISTATS 2017.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import flwr as fl
    from flwr.common import FitIns, FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.client_manager import ClientManager
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import FedAvg as FlowerFedAvg

    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    fl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class FedAvgStrategy:
    """FedAvg strategy wrapping Flower's FedAvg implementation.

    Falls back to a pure-NumPy weighted average when Flower is not available
    (e.g. in unit tests with mocked dependencies).
    """

    strategy_name = "fedavg"

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 0,
        min_available_clients: int = 2,
    ) -> None:
        self._fraction_fit = fraction_fit
        self._fraction_evaluate = fraction_evaluate
        self._min_fit_clients = min_fit_clients
        self._min_evaluate_clients = min_evaluate_clients
        self._min_available_clients = min_available_clients

        if FLOWER_AVAILABLE:
            self._flower_strategy = FlowerFedAvg(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
            )
        else:
            self._flower_strategy = None
            logger.warning("Flower not available — FedAvgStrategy running in fallback mode")

    def configure_fit(
        self,
        server_round: int,
        parameters: list[np.ndarray[Any, Any]],
        client_manager: Any,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Configure the next fit round."""
        if self._flower_strategy is not None and FLOWER_AVAILABLE:
            flower_params = ndarrays_to_parameters(parameters)
            fit_configs = self._flower_strategy.configure_fit(
                server_round=server_round,
                parameters=flower_params,
                client_manager=client_manager,
            )
            return [(client, {"server_round": server_round}) for client, _ in fit_configs]
        # Fallback: return empty list (caller must handle client selection)
        return []

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[Any, Any]],
        failures: list[Any],
    ) -> tuple[list[np.ndarray[Any, Any]] | None, dict[str, Any]]:
        """Aggregate fit results using weighted averaging."""
        if not results:
            return None, {}

        if self._flower_strategy is not None and FLOWER_AVAILABLE:
            aggregated_params, metrics = self._flower_strategy.aggregate_fit(
                server_round=server_round,
                results=results,
                failures=failures,
            )
            if aggregated_params is None:
                return None, metrics
            return parameters_to_ndarrays(aggregated_params), metrics

        # Pure-NumPy fallback
        return self._numpy_fedavg(results), {"aggregation_method": "fedavg_numpy_fallback"}

    def _numpy_fedavg(
        self,
        results: list[tuple[Any, Any]],
    ) -> list[np.ndarray[Any, Any]]:
        """Weighted average of numpy parameter arrays."""
        # results: list of (client_proxy, fit_res) — fit_res has .parameters and .num_examples
        weights_results: list[tuple[list[np.ndarray[Any, Any]], int]] = []
        for _, fit_res in results:
            params = fit_res.parameters if hasattr(fit_res, "parameters") else fit_res[0]
            num_examples = fit_res.num_examples if hasattr(fit_res, "num_examples") else fit_res[1]
            if FLOWER_AVAILABLE:
                ndarrays = parameters_to_ndarrays(params)
            else:
                ndarrays = params
            weights_results.append((ndarrays, num_examples))

        total_examples = sum(n for _, n in weights_results)
        averaged: list[np.ndarray[Any, Any]] = []
        for layer_idx in range(len(weights_results[0][0])):
            weighted_sum = np.zeros_like(weights_results[0][0][layer_idx], dtype=np.float64)
            for params, num_examples in weights_results:
                weighted_sum += params[layer_idx] * (num_examples / total_examples)
            averaged.append(weighted_sum)
        return averaged

    def configure_evaluate(
        self,
        server_round: int,
        parameters: list[np.ndarray[Any, Any]],
        client_manager: Any,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Configure distributed evaluation (disabled by default)."""
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
        total_examples = sum(r[1].num_examples for r in results if hasattr(r[1], "num_examples"))
        if total_examples == 0:
            return None, {}
        weighted_loss = sum(
            r[1].loss * r[1].num_examples / total_examples
            for r in results
            if hasattr(r[1], "num_examples")
        )
        return weighted_loss, {"num_clients_evaluated": len(results)}
