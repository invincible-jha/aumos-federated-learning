"""AumOS Federated Learning client for TensorFlow/Keras models.

Gap #145: Direct ML Framework Integration — TensorFlow/Keras.

Wraps a tf.keras.Model so ML engineers interact only with familiar Keras APIs.
All Flower NumPy serialization is handled transparently.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class AumOSTFFlowerClient:
    """AumOS FL client wrapping a tf.keras.Model via the Flower NumPyClient protocol.

    Equivalent to AumOSFlowerClient but for TensorFlow/Keras models.
    Handles weight serialization and Flower communication automatically.

    Example::

        model = tf.keras.Sequential([...])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        client = AumOSTFFlowerClient(
            model=model,
            train_data=train_ds,
            test_data=test_ds,
            job_id="fed-job-uuid",
            participant_token="my-token",
        )
        client.start(server_address="fl.aumos.ai:8080")

    Args:
        model: A compiled ``tf.keras.Model`` to train federally.
        train_data: ``tf.data.Dataset`` for local training.
        test_data: ``tf.data.Dataset`` for local evaluation.
        job_id: AumOS FL job UUID (from the server).
        participant_token: Token issued by ``POST /fl/jobs/{id}/join``.
        local_epochs: Number of local training epochs per FL round.
    """

    def __init__(
        self,
        model: Any,
        train_data: Any,
        test_data: Any,
        job_id: str,
        participant_token: str,
        local_epochs: int = 1,
    ) -> None:
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.job_id = job_id
        self.participant_token = participant_token
        self.local_epochs = local_epochs

    # ------------------------------------------------------------------
    # Flower NumPyClient interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray[Any, Any]]:
        """Extract current Keras model weights as NumPy arrays.

        Args:
            config: Server configuration (unused by this implementation).

        Returns:
            List of NumPy weight arrays from ``model.get_weights()``.
        """
        return self.model.get_weights()

    def set_parameters(self, parameters: list[np.ndarray[Any, Any]]) -> None:
        """Load server-aggregated weights into the local model.

        Args:
            parameters: List of NumPy arrays from ``model.set_weights()``.
        """
        self.model.set_weights(parameters)

    def fit(
        self,
        parameters: list[np.ndarray[Any, Any]],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray[Any, Any]], int, dict[str, Any]]:
        """Run one round of local Keras training.

        Args:
            parameters: Global model weights from the server.
            config: Round configuration dict (may override ``local_epochs``).

        Returns:
            Tuple of (updated_parameters, num_examples, metrics_dict).
        """
        self.set_parameters(parameters)
        local_epochs = int(config.get("local_epochs", self.local_epochs))
        history = self.model.fit(self.train_data, epochs=local_epochs, verbose=0)

        num_examples = sum(1 for _ in self.train_data)
        avg_loss = float(history.history.get("loss", [0.0])[-1])

        logger.info(
            "Local Keras training complete",
            job_id=self.job_id,
            local_epochs=local_epochs,
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
        results = self.model.evaluate(self.test_data, verbose=0, return_dict=True)
        loss = float(results.get("loss", 0.0))
        accuracy = float(results.get("accuracy", 0.0))
        num_examples = sum(1 for _ in self.test_data)
        return loss, num_examples, {"accuracy": accuracy}

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
            def __init__(self, outer: AumOSTFFlowerClient) -> None:
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
            "Starting TF Flower client",
            server_address=server_address,
            ssl=ssl,
            job_id=self.job_id,
        )
        fl.client.start_client(
            server_address=server_address,
            client=_NumPyWrapper(self).to_client(),
        )
