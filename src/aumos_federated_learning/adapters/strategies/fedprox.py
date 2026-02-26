"""FedProx strategy — Federated Optimization with proximal term.

FedProx adds a proximal regularization term μ/2 * ||w - w_t||² to each
client's local objective, limiting how far client weights deviate from the
global model. This improves stability on heterogeneous (non-IID) data.

Reference: Li et al., "Federated Optimization in Heterogeneous Networks",
MLSys 2020.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.strategy import FedProx as FlowerFedProx

    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    fl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class FedProxStrategy:
    """FedProx strategy wrapping Flower's FedProx implementation.

    The proximal_mu parameter controls the strength of regularization:
    - mu=0.0 is equivalent to FedAvg
    - Larger mu → more conservative local updates → slower but more stable convergence
    """

    strategy_name = "fedprox"

    def __init__(
        self,
        *,
        proximal_mu: float = 0.1,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 0,
        min_available_clients: int = 2,
    ) -> None:
        self.proximal_mu = proximal_mu
        self._fraction_fit = fraction_fit
        self._min_fit_clients = min_fit_clients
        self._min_available_clients = min_available_clients

        if FLOWER_AVAILABLE:
            self._flower_strategy = FlowerFedProx(
                proximal_mu=proximal_mu,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
            )
        else:
            self._flower_strategy = None
            logger.warning("Flower not available — FedProxStrategy running in fallback mode")

    def configure_fit(
        self,
        server_round: int,
        parameters: list[np.ndarray[Any, Any]],
        client_manager: Any,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Configure the next fit round, passing proximal_mu to clients."""
        if self._flower_strategy is not None and FLOWER_AVAILABLE:
            flower_params = ndarrays_to_parameters(parameters)
            fit_configs = self._flower_strategy.configure_fit(
                server_round=server_round,
                parameters=flower_params,
                client_manager=client_manager,
            )
            return [
                (client, {"server_round": server_round, "proximal_mu": self.proximal_mu})
                for client, _ in fit_configs
            ]
        return []

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[Any, Any]],
        failures: list[Any],
    ) -> tuple[list[np.ndarray[Any, Any]] | None, dict[str, Any]]:
        """Aggregate using weighted average (same as FedAvg — proximal term is client-side)."""
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
            return parameters_to_ndarrays(aggregated_params), {
                **metrics,
                "proximal_mu": self.proximal_mu,
            }

        return None, {"proximal_mu": self.proximal_mu}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: list[np.ndarray[Any, Any]],
        client_manager: Any,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Configure evaluation round."""
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
        return weighted_loss, {"proximal_mu": self.proximal_mu}
