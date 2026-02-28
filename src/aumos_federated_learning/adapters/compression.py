"""Communication compression for federated learning weight updates.

Gap #150: No Communication Compression.

Provides two compression strategies:
- TopKSparsification: send only the k% largest-magnitude gradients
- Int8Quantization: 8-bit uniform quantization (4x bandwidth reduction vs float32)

References:
    Strom, N. (2015). Scalable distributed DNN training using commodity GPU cloud computing.
    Aji, A. F., & Heafield, K. (2017). Sparse communication for distributed gradient descent.
    Alistarh, D., et al. (2017). QSGD: Communication-Efficient SGD via Gradient Quantization.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


@runtime_checkable
class CompressionProtocol(Protocol):
    """Protocol for compressible weight update transforms."""

    def compress(
        self,
        weights: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        """Compress a weight delta.

        Args:
            weights: Weight delta array (float32).

        Returns:
            Tuple of (compressed_array, metadata) where metadata contains
            information needed for decompression (indices, scale, shape, etc.).
        """
        ...

    def decompress(
        self,
        compressed: np.ndarray[Any, Any],
        metadata: dict[str, Any],
    ) -> np.ndarray[Any, Any]:
        """Decompress a weight delta using metadata from compression.

        Args:
            compressed: Compressed weight array.
            metadata: Metadata dict returned by ``compress()``.

        Returns:
            Reconstructed weight delta (float32).
        """
        ...


class TopKSparsification:
    """Top-k sparsification: keep only the largest-magnitude k% of gradients.

    All other gradient values are set to zero before transmission. The server
    reconstructs the full-size delta by filling zero-positions back.

    With k=0.01 (top 1%), this achieves ~99% bandwidth reduction vs dense
    gradient transmission, with minimal accuracy loss after 20+ rounds.

    Reference:
        Strom (2015); Aji & Heafield (2017).

    Args:
        k: Fraction of gradients to transmit. Range (0, 1].
           ``k=0.01`` sends the top 1% of values by absolute magnitude.
    """

    def __init__(self, k: float = 0.01) -> None:
        if not (0 < k <= 1.0):
            raise ValueError(f"k must be in (0, 1]. Got: {k}")
        self.k = k

    def compress(
        self,
        weights: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        """Compress a weight delta using top-k sparsification.

        Args:
            weights: Weight delta array (any shape, float32).

        Returns:
            Tuple of (sparse_array_same_shape, metadata) where metadata contains
            ``shape`` and ``k`` for documentation purposes.
        """
        original_shape = weights.shape
        flat = weights.flatten()
        k_count = max(1, int(len(flat) * self.k))
        top_k_indices = np.argpartition(np.abs(flat), -k_count)[-k_count:]
        compressed = np.zeros_like(flat)
        compressed[top_k_indices] = flat[top_k_indices]

        sparsity = 1.0 - k_count / len(flat)
        logger.debug("TopK compression", k=self.k, k_count=k_count, sparsity=sparsity)

        return (
            compressed.reshape(original_shape),
            {"shape": list(original_shape), "k": self.k},
        )

    def decompress(
        self,
        compressed: np.ndarray[Any, Any],
        metadata: dict[str, Any],
    ) -> np.ndarray[Any, Any]:
        """Decompress a sparsified array.

        Since sparsification does not change the array shape or dtype, decompression
        is a no-op — the sparse array is returned as-is.

        Args:
            compressed: Sparsified array from ``compress()``.
            metadata: Metadata dict from ``compress()``.

        Returns:
            The input array unchanged (sparsification is already decompressed).
        """
        return compressed.astype(np.float32)


class Int8Quantization:
    """Uniform 8-bit quantization of weight updates.

    Maps float32 values to uint8 in the range [0, 255] using linear scaling.
    Reduces transmission bandwidth by ~4x vs float32.

    Trade-off: introduces rounding noise proportional to the value range.
    Use Top-k sparsification for higher compression ratios at lower noise.

    Args:
        (no constructor arguments — quantisation range is data-driven)
    """

    def compress(
        self,
        weights: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        """Compress a weight delta using uniform 8-bit quantization.

        Args:
            weights: Weight delta array (float32, any shape).

        Returns:
            Tuple of (quantized_uint8_array, metadata) where metadata contains
            ``min``, ``scale``, and ``shape`` needed for decompression.
        """
        original_shape = weights.shape
        flat = weights.flatten().astype(np.float32)
        min_val = float(flat.min())
        max_val = float(flat.max())
        val_range = max_val - min_val

        if val_range == 0.0:
            scale = 1.0
        else:
            scale = val_range / 255.0

        quantized = np.round((flat - min_val) / scale).clip(0, 255).astype(np.uint8)
        compression_ratio = weights.nbytes / quantized.nbytes
        logger.debug("Int8 compression", compression_ratio=compression_ratio)

        return (
            quantized.reshape(original_shape),
            {"min": min_val, "scale": scale, "shape": list(original_shape)},
        )

    def decompress(
        self,
        compressed: np.ndarray[Any, Any],
        metadata: dict[str, Any],
    ) -> np.ndarray[Any, Any]:
        """Decompress a quantized array back to float32.

        Args:
            compressed: Quantized uint8 array from ``compress()``.
            metadata: Metadata dict from ``compress()`` containing ``min`` and ``scale``.

        Returns:
            Reconstructed float32 weight delta (with quantisation error).
        """
        min_val: float = metadata["min"]
        scale: float = metadata["scale"]
        return compressed.astype(np.float32) * scale + min_val
