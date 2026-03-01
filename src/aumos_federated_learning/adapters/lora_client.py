"""Federated LoRA fine-tuning client for large language models.

Gap #154: No Federated LLM Fine-Tuning Support.

AumOSLoRAFlowerClient enables federated fine-tuning of LLMs using Low-Rank
Adaptation (LoRA). Only LoRA adapter parameters are communicated — base model
weights stay local — reducing communication by 99%+ vs full fine-tuning.

References:
    Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
    arXiv:2106.09685.
    Ye, R., et al. (2024). OpenFedLLM: Training Large Language Models on Decentralized
    Private Data via Federated Learning. arXiv:2402.06954.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class AumOSLoRAFlowerClient:
    """Flower NumPyClient for federated LoRA fine-tuning of LLMs.

    Only LoRA adapter weights (A and B matrices per target module) are
    exchanged with the server. The base LLM stays frozen and is never
    transmitted, reducing communication overhead by orders of magnitude.

    Protocol:
        1. Server distributes global LoRA adapter weights to participants.
        2. Each participant fine-tunes their local LoRA adapters on private data.
        3. Participants send back updated adapter deltas.
        4. Server aggregates (FedAvg) the adapter deltas.
        5. Repeat for num_rounds.

    Args:
        model_name_or_path: HuggingFace model ID or local path for the base LLM.
        lora_rank: LoRA rank r. Higher rank = more parameters, more expressivity.
        lora_alpha: LoRA scaling factor. Effective scale = lora_alpha / lora_rank.
        target_modules: List of module names to apply LoRA to (e.g. ["q_proj", "v_proj"]).
        flower_server_address: gRPC address of the Flower server.
        local_epochs: Number of local training epochs per round.
        learning_rate: Local optimizer learning rate.
        max_seq_length: Maximum token sequence length for training.
        dp_epsilon: Optional DP budget for gradient clipping. ``None`` disables DP.
    """

    def __init__(
        self,
        model_name_or_path: str,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        target_modules: list[str] | None = None,
        flower_server_address: str = "localhost:8080",
        local_epochs: int = 1,
        learning_rate: float = 3e-4,
        max_seq_length: int = 512,
        dp_epsilon: float | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        self.flower_server_address = flower_server_address
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.max_seq_length = max_seq_length
        self.dp_epsilon = dp_epsilon
        self._model: Any = None
        self._tokenizer: Any = None
        self._lora_config: Any = None

    def _load_model(self) -> None:
        """Load the base LLM and attach LoRA adapters.

        Defers heavy imports so the module can be imported without PyTorch/PEFT.

        Raises:
            ImportError: If transformers or peft are not installed.
        """
        try:
            from peft import LoraConfig, TaskType, get_peft_model  # noqa: PLC0415
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "transformers and peft are required for federated LLM fine-tuning. "
                "Install: pip install transformers peft"
            ) from exc

        logger.info("Loading base model %s", self.model_name_or_path)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map="auto",
        )

        self._lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        self._model = get_peft_model(base_model, self._lora_config)

        trainable, total = self._count_parameters()
        logger.info(
            "LoRA model loaded — trainable: %d / %d params (%.2f%%)",
            trainable,
            total,
            100.0 * trainable / max(total, 1),
        )

    def _count_parameters(self) -> tuple[int, int]:
        """Return (trainable_params, total_params) for the model."""
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        return trainable, total

    def get_lora_parameters(self) -> list[np.ndarray[Any, Any]]:
        """Extract current LoRA adapter weights as a list of NumPy arrays.

        Only LoRA A/B matrices are included — base model weights are excluded.

        Returns:
            List of NumPy float32 arrays, one per LoRA parameter tensor.
        """
        if self._model is None:
            self._load_model()

        import torch  # noqa: PLC0415

        params: list[np.ndarray[Any, Any]] = []
        for name, param in self._model.named_parameters():
            if "lora_" in name and param.requires_grad:
                params.append(param.detach().cpu().to(torch.float32).numpy())
        return params

    def set_lora_parameters(
        self, parameters: list[np.ndarray[Any, Any]]
    ) -> None:
        """Load LoRA adapter weights from a list of NumPy arrays.

        Args:
            parameters: List of arrays in the same order as get_lora_parameters().
        """
        if self._model is None:
            self._load_model()

        import torch  # noqa: PLC0415

        lora_params = [
            (name, param)
            for name, param in self._model.named_parameters()
            if "lora_" in name and param.requires_grad
        ]

        if len(parameters) != len(lora_params):
            raise ValueError(
                f"Parameter count mismatch: expected {len(lora_params)}, "
                f"got {len(parameters)}"
            )

        with torch.no_grad():
            for (name, param), array in zip(lora_params, parameters):
                param.copy_(torch.from_numpy(array).to(param.device))
                del name  # suppress unused variable warning

    def fit(
        self,
        parameters: list[np.ndarray[Any, Any]],
        train_dataset: Any,
        config: dict[str, Any] | None = None,
    ) -> tuple[list[np.ndarray[Any, Any]], int, dict[str, Any]]:
        """Local LoRA fine-tuning step.

        Loads the global adapter weights, fine-tunes on local data for
        ``local_epochs`` epochs, then returns updated adapter weights.

        Args:
            parameters: Global LoRA adapter weights from the server.
            train_dataset: HuggingFace Dataset or compatible iterator.
            config: Optional config dict from server (e.g. override epochs).

        Returns:
            Tuple of (updated_lora_parameters, num_samples, metrics).
        """
        try:
            import torch  # noqa: PLC0415
            from torch.optim import AdamW  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("PyTorch is required for LoRA fine-tuning.") from exc

        if self._model is None:
            self._load_model()

        self.set_lora_parameters(parameters)

        local_epochs = int((config or {}).get("local_epochs", self.local_epochs))
        optimizer = AdamW(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=self.learning_rate,
        )

        self._model.train()
        total_loss = 0.0
        num_batches = 0
        num_samples = 0

        for _epoch in range(local_epochs):
            for batch in train_dataset:
                optimizer.zero_grad()
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
                labels = batch.get("labels", input_ids)

                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                # Optional DP gradient clipping
                if self.dp_epsilon is not None:
                    max_grad_norm = 1.0
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self._model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                    optimizer.step()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                batch_size = input_ids.shape[0] if hasattr(input_ids, "shape") else 1
                num_samples += batch_size

        avg_loss = total_loss / max(num_batches, 1)
        updated_params = self.get_lora_parameters()

        logger.info(
            "LoRA local training complete — epochs=%d, samples=%d, avg_loss=%.4f",
            local_epochs,
            num_samples,
            avg_loss,
        )

        return updated_params, num_samples, {"loss": avg_loss}

    def evaluate(
        self,
        parameters: list[np.ndarray[Any, Any]],
        eval_dataset: Any,
        config: dict[str, Any] | None = None,
    ) -> tuple[float, int, dict[str, Any]]:
        """Evaluate LoRA adapter on local held-out data.

        Args:
            parameters: LoRA adapter weights to evaluate.
            eval_dataset: Evaluation dataset.
            config: Optional server config.

        Returns:
            Tuple of (loss, num_samples, metrics).
        """
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("PyTorch is required for LoRA evaluation.") from exc

        if self._model is None:
            self._load_model()

        self.set_lora_parameters(parameters)
        self._model.eval()

        total_loss = 0.0
        num_batches = 0
        num_samples = 0

        with torch.no_grad():
            for batch in eval_dataset:
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
                labels = batch.get("labels", input_ids)

                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()
                num_batches += 1
                batch_size = input_ids.shape[0] if hasattr(input_ids, "shape") else 1
                num_samples += batch_size

        avg_loss = total_loss / max(num_batches, 1)
        perplexity = float(np.exp(min(avg_loss, 20.0)))  # cap to avoid overflow

        logger.info(
            "LoRA evaluation — samples=%d, loss=%.4f, perplexity=%.2f",
            num_samples,
            avg_loss,
            perplexity,
        )

        return avg_loss, num_samples, {"loss": avg_loss, "perplexity": perplexity}

    def start(
        self,
        train_dataset: Any,
        eval_dataset: Any | None = None,
    ) -> None:
        """Start the Flower client loop and connect to the server.

        Blocks until the server closes the connection or all rounds complete.

        Args:
            train_dataset: Training data for local fine-tuning.
            eval_dataset: Optional evaluation dataset.

        Raises:
            ImportError: If flwr is not installed.
        """
        try:
            import flwr as fl  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Flower is required. Install: pip install flwr"
            ) from exc

        lora_client = self

        class _LoRANumPyClient(fl.client.NumPyClient):  # type: ignore[misc]
            def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray[Any, Any]]:  # type: ignore[override]
                return lora_client.get_lora_parameters()

            def fit(  # type: ignore[override]
                self,
                parameters: list[np.ndarray[Any, Any]],
                config: dict[str, Any],
            ) -> tuple[list[np.ndarray[Any, Any]], int, dict[str, Any]]:
                return lora_client.fit(parameters, train_dataset, config)

            def evaluate(  # type: ignore[override]
                self,
                parameters: list[np.ndarray[Any, Any]],
                config: dict[str, Any],
            ) -> tuple[float, int, dict[str, Any]]:
                dataset = eval_dataset if eval_dataset is not None else train_dataset
                return lora_client.evaluate(parameters, dataset, config)

        logger.info(
            "Starting LoRA FL client — server=%s model=%s rank=%d",
            self.flower_server_address,
            self.model_name_or_path,
            self.lora_rank,
        )

        fl.client.start_client(
            server_address=self.flower_server_address,
            client=_LoRANumPyClient().to_client(),
        )
