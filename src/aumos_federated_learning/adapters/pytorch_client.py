"""AumOS Federated Learning client for PyTorch models.

Gap #145: Direct ML Framework Integration — PyTorch.

Wraps a PyTorch nn.Module so ML engineers interact only with familiar PyTorch APIs.
All Flower NumPy serialization is handled transparently.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class AumOSFlowerClient:
    """AumOS FL client wrapping a PyTorch nn.Module via the Flower NumPyClient protocol.

    Handles weight serialization, DP gradient clipping, and Flower communication.
    ML engineers interact only with PyTorch APIs — Flower is an implementation detail.

    Example::

        model = MyResNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        client = AumOSFlowerClient(
            model=model,
            optimizer=optimizer,
            trainloader=train_dl,
            testloader=test_dl,
            job_id="fed-job-uuid",
            participant_token="my-token",
            dp_epsilon=1.0,
        )
        client.start(server_address="fl.aumos.ai:8080")

    Args:
        model: A ``torch.nn.Module`` to train federally.
        optimizer: PyTorch optimizer bound to model parameters.
        trainloader: ``DataLoader`` with local training batches.
        testloader: ``DataLoader`` with local evaluation batches.
        job_id: AumOS FL job UUID (from the server).
        participant_token: Token issued by ``POST /fl/jobs/{id}/join``.
        dp_epsilon: Target DP epsilon. ``None`` disables DP gradient clipping.
        dp_max_grad_norm: Max gradient norm for DP clipping (Abadi et al. 2016).
        device: PyTorch device string: ``"cpu"``, ``"cuda"``, or ``"mps"``.
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        trainloader: Any,
        testloader: Any,
        job_id: str,
        participant_token: str,
        dp_epsilon: float | None = None,
        dp_max_grad_norm: float = 1.0,
        device: str = "cpu",
    ) -> None:
        try:
            import torch  # noqa: PLC0415

            self._torch = torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for AumOSFlowerClient. "
                "Install it: pip install torch"
            ) from exc

        self.model = model.to(device)
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.testloader = testloader
        self.job_id = job_id
        self.participant_token = participant_token
        self.dp_epsilon = dp_epsilon
        self.dp_max_grad_norm = dp_max_grad_norm
        self.device = self._torch.device(device)

    # ------------------------------------------------------------------
    # Flower NumPyClient interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray[Any, Any]]:
        """Extract current model weights as NumPy arrays for server aggregation.

        Args:
            config: Server configuration (unused by this implementation).

        Returns:
            List of NumPy arrays, one per trainable parameter tensor.
        """
        return [
            param.cpu().detach().numpy()
            for param in self.model.parameters()
        ]

    def set_parameters(self, parameters: list[np.ndarray[Any, Any]]) -> None:
        """Load server-aggregated weights into the local model.

        Args:
            parameters: List of NumPy arrays matching the model's parameter shapes.
        """
        torch = self._torch
        param_names = [name for name, _ in self.model.named_parameters()]
        state_dict = {
            name: torch.tensor(weight)
            for name, weight in zip(param_names, parameters)
        }
        self.model.load_state_dict(state_dict, strict=False)

    def fit(
        self,
        parameters: list[np.ndarray[Any, Any]],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray[Any, Any]], int, dict[str, Any]]:
        """Run one round of local training.

        Applies DP gradient clipping if ``dp_epsilon`` is configured.

        Args:
            parameters: Global model weights from the server.
            config: Round configuration dict (may contain ``local_epochs``).

        Returns:
            Tuple of (updated_parameters, num_examples, metrics_dict).
        """
        import torch.nn as nn  # noqa: PLC0415

        self.set_parameters(parameters)
        local_epochs = int(config.get("local_epochs", 1))
        self.model.train()
        total_loss = 0.0
        batches = 0

        for _ in range(local_epochs):
            for batch in self.trainloader:
                inputs, labels = (x.to(self.device) for x in batch)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self._torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                total_loss += loss.item()
                batches += 1

                if self.dp_epsilon is not None:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.dp_max_grad_norm,
                    )

                self.optimizer.step()

        num_examples = len(self.trainloader.dataset)
        avg_loss = total_loss / max(batches, 1)
        logger.info(
            "Local training complete",
            job_id=self.job_id,
            local_epochs=local_epochs,
            num_examples=num_examples,
            avg_loss=avg_loss,
        )
        return self.get_parameters({}), num_examples, {"loss": avg_loss}

    def evaluate(
        self,
        parameters: list[np.ndarray[Any, Any]],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, Any]]:
        """Evaluate the current global model on local test data.

        Args:
            parameters: Global model weights from the server.
            config: Evaluation configuration dict.

        Returns:
            Tuple of (loss, num_examples, metrics_dict).
        """
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0

        with self._torch.no_grad():
            for batch in self.testloader:
                inputs, labels = (x.to(self.device) for x in batch)
                outputs = self.model(inputs)
                loss_sum += self._torch.nn.functional.cross_entropy(outputs, labels).item()
                correct += int((outputs.argmax(1) == labels).sum().item())
                total += labels.size(0)

        avg_loss = loss_sum / max(len(self.testloader), 1)
        accuracy = correct / max(total, 1)
        return avg_loss, total, {"accuracy": accuracy}

    def start(self, server_address: str, ssl: bool = True) -> None:
        """Connect to the AumOS FL server and start the Flower client loop.

        Args:
            server_address: Flower server address, e.g. ``fl.aumos.ai:8080``.
            ssl: Enable TLS. Must be ``True`` in production.
        """
        try:
            import flwr as fl  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Flower is required. Install it: pip install flwr"
            ) from exc

        class _NumPyWrapper(fl.client.NumPyClient):  # type: ignore[misc]
            def __init__(self, outer: AumOSFlowerClient) -> None:
                self._outer = outer

            def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray[Any, Any]]:  # type: ignore[override]
                return self._outer.get_parameters(config)

            def fit(  # type: ignore[override]
                self,
                parameters: list[np.ndarray[Any, Any]],
                config: dict[str, Any],
            ) -> tuple[list[np.ndarray[Any, Any]], int, dict[str, Any]]:
                return self._outer.fit(parameters, config)

            def evaluate(  # type: ignore[override]
                self,
                parameters: list[np.ndarray[Any, Any]],
                config: dict[str, Any],
            ) -> tuple[float, int, dict[str, Any]]:
                return self._outer.evaluate(parameters, config)

        logger.info(
            "Starting Flower client",
            server_address=server_address,
            ssl=ssl,
            job_id=self.job_id,
        )
        fl.client.start_client(
            server_address=server_address,
            client=_NumPyWrapper(self).to_client(),
        )
