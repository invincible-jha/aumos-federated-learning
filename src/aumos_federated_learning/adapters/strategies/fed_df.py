"""FedDF: Federated Dataset Distillation strategy for heterogeneous model architectures.

Gap #151: No Heterogeneous Model Support.

Instead of averaging model weights (which requires identical architectures), FedDF
aggregates logits on a small public unlabeled dataset via ensemble distillation.

Reference:
    Lin, T., Kong, L., Stich, S. U., & Jaggi, M. (2020).
    Ensemble Distillation for Robust Model Fusion in Federated Learning.
    arXiv:2006.07242.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class FedDFStrategy:
    """Knowledge distillation strategy for heterogeneous FL.

    Supports participants with different model architectures by aggregating
    logits rather than weights. Protocol:

    1. Each participant trains locally on private data (any architecture).
    2. Each participant computes logits on a shared public dataset and sends them.
    3. The server ensembles the logits (simple average or weighted average).
    4. The server trains a global model on the ensembled logits via distillation.
    5. Participants receive the global model logits for the next round.

    The public dataset is small (default 1,000 samples) and contains no private data.
    MNIST/CIFAR test splits are typical choices.

    Args:
        num_classes: Number of output classes for the distillation task.
        public_dataset_size: Number of public samples used for distillation.
        distillation_epochs: Epochs to train the server model on ensembled logits.
        temperature: Distillation temperature. Higher values produce softer targets.
        min_participants: Minimum number of logit submissions before ensembling.
    """

    strategy_name: str = "fed_df"

    def __init__(
        self,
        num_classes: int = 10,
        public_dataset_size: int = 1000,
        distillation_epochs: int = 5,
        temperature: float = 1.0,
        min_participants: int = 2,
    ) -> None:
        self.num_classes = num_classes
        self.public_dataset_size = public_dataset_size
        self.distillation_epochs = distillation_epochs
        self.temperature = temperature
        self.min_participants = min_participants
        # Buffer: participant_id → logits array (public_dataset_size, num_classes)
        self._logit_buffer: dict[str, np.ndarray[Any, Any]] = {}
        self._current_round: int = 0
        # Global ensembled logits (shared with participants as soft labels)
        self._global_logits: np.ndarray[Any, Any] | None = None

    def submit_logits(
        self,
        participant_id: str,
        logits: np.ndarray[Any, Any],
    ) -> None:
        """Accept a participant's logit contribution for the current round.

        Args:
            participant_id: Identifier of the submitting participant.
            logits: Array of shape ``(public_dataset_size, num_classes)``.
        """
        if logits.shape != (self.public_dataset_size, self.num_classes):
            raise ValueError(
                f"Expected logits shape ({self.public_dataset_size}, {self.num_classes}), "
                f"got {logits.shape}"
            )
        self._logit_buffer[participant_id] = logits
        logger.debug(
            "Logits received",
            participant_id=participant_id,
            round=self._current_round,
            buffer_size=len(self._logit_buffer),
        )

    def ensemble_logits(self) -> np.ndarray[Any, Any] | None:
        """Ensemble buffered logits when enough participants have submitted.

        Returns:
            Ensembled logit array of shape ``(public_dataset_size, num_classes)``,
            or ``None`` if fewer than ``min_participants`` have submitted.
        """
        if len(self._logit_buffer) < self.min_participants:
            return None

        stacked = np.stack(list(self._logit_buffer.values()), axis=0)
        ensembled = stacked.mean(axis=0)

        # Apply temperature scaling to soften targets for distillation
        if self.temperature != 1.0:
            ensembled = ensembled / self.temperature

        self._global_logits = ensembled
        self._logit_buffer.clear()
        self._current_round += 1

        logger.info(
            "Logit ensembling complete",
            round=self._current_round,
            num_participants=len(stacked),
        )
        return ensembled

    def get_soft_labels(self) -> np.ndarray[Any, Any] | None:
        """Return the most recent ensembled soft labels for client-side distillation.

        Returns:
            Soft label array ``(public_dataset_size, num_classes)``, or ``None``
            if no ensembling has occurred yet.
        """
        return self._global_logits

    def knowledge_distillation_loss(
        self,
        student_logits: np.ndarray[Any, Any],
        teacher_logits: np.ndarray[Any, Any],
    ) -> float:
        """Compute KL-divergence loss between student and teacher soft labels (NumPy).

        This is a pure-NumPy reference implementation for testing. In production,
        use the equivalent PyTorch or TensorFlow loss function with GPU acceleration.

        Args:
            student_logits: Student model logits, shape ``(N, C)``.
            teacher_logits: Teacher ensemble logits, shape ``(N, C)``.

        Returns:
            Scalar KL-divergence loss value.
        """
        # Softmax normalization
        def softmax(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
            exp_x = np.exp(x - x.max(1, keepdims=True))
            return exp_x / exp_x.sum(1, keepdims=True)

        p_teacher = softmax(teacher_logits / self.temperature)
        p_student = softmax(student_logits / self.temperature)
        # KL(teacher || student)
        eps = 1e-8
        kl = float(np.sum(p_teacher * np.log((p_teacher + eps) / (p_student + eps)), axis=1).mean())
        return kl * self.temperature ** 2
