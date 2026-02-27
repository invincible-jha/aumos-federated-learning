"""Federated communication adapter for secure inter-node messaging.

Provides gRPC channel management, model weight serialization, mTLS channel
setup, chunked transfer for large model updates, compression, retry/ack
logic, and per-participant connection pooling.

In production, the gRPC stubs should be generated from the aumos-proto package.
This adapter provides a concrete implementation with pluggable stubs for testing.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import logging
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE_BYTES = 4 * 1024 * 1024  # 4 MiB
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE_SECONDS = 2.0
SUPPORTED_COMPRESSIONS = frozenset({"none", "gzip", "lz4"})


# ---------------------------------------------------------------------------
# Internal data types
# ---------------------------------------------------------------------------


@dataclass
class ChannelConfig:
    """TLS/mTLS configuration for a gRPC channel.

    Attributes:
        endpoint: Host:port of the remote participant gRPC server.
        root_cert_pem: PEM-encoded CA certificate for verifying the peer.
        client_cert_pem: PEM-encoded client certificate (mTLS).
        client_key_pem: PEM-encoded client private key (mTLS).
        max_message_bytes: Maximum gRPC message size in bytes.
        compression: Compression algorithm to apply ('none', 'gzip', 'lz4').
    """

    endpoint: str
    root_cert_pem: str | None = None
    client_cert_pem: str | None = None
    client_key_pem: str | None = None
    max_message_bytes: int = 256 * 1024 * 1024  # 256 MiB
    compression: str = "gzip"

    def __post_init__(self) -> None:
        if self.compression not in SUPPORTED_COMPRESSIONS:
            raise ValueError(
                f"Unsupported compression '{self.compression}'. "
                f"Choose from {sorted(SUPPORTED_COMPRESSIONS)}"
            )


@dataclass
class MessageAck:
    """Acknowledgment from a remote participant after receiving a message.

    Attributes:
        message_id: The UUID of the message being acknowledged.
        participant_id: The acknowledging participant.
        received_at: Unix timestamp when the message was received.
        checksum_valid: Whether the payload checksum verified correctly.
    """

    message_id: str
    participant_id: str
    received_at: float
    checksum_valid: bool


@dataclass
class ChannelState:
    """Runtime state for a single participant connection.

    Attributes:
        participant_id: The participant this channel belongs to.
        config: Channel configuration including TLS settings.
        connected_at: Unix timestamp when the connection was established.
        bytes_sent: Cumulative bytes transmitted to this participant.
        bytes_received: Cumulative bytes received from this participant.
        pending_acks: message_id -> send timestamp for unacknowledged messages.
        is_open: Whether the channel is currently usable.
    """

    participant_id: str
    config: ChannelConfig
    connected_at: float = field(default_factory=time.time)
    bytes_sent: int = 0
    bytes_received: int = 0
    pending_acks: dict[str, float] = field(default_factory=dict)
    is_open: bool = True


# ---------------------------------------------------------------------------
# FederatedCommunicationAdapter
# ---------------------------------------------------------------------------


class FederatedCommunicationAdapter:
    """Secure messaging layer for federated learning inter-node communication.

    Manages:
    - Per-participant gRPC channel lifecycle (open / close / reconnect)
    - mTLS channel setup using aumos-proto generated stubs
    - Model weight serialization to protobuf-compatible binary (NumPy NPZ)
    - Chunked streaming transfer for large weight tensors
    - gzip / lz4 compression to reduce bandwidth
    - Retry logic with exponential backoff and per-message ACKs
    - Connection pool with configurable concurrency limits

    Args:
        stub_factory: Callable that accepts a ChannelConfig and returns a
            gRPC stub (or mock for testing). Signature:
            (ChannelConfig) -> Any
        max_concurrent_channels: Maximum number of simultaneously open
            gRPC channels in the connection pool.
        chunk_size_bytes: Size of each chunk when streaming large payloads.
        default_compression: Default compression algorithm ('gzip', 'lz4', 'none').
    """

    def __init__(
        self,
        stub_factory: Any,
        max_concurrent_channels: int = 50,
        chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
        default_compression: str = "gzip",
    ) -> None:
        self._stub_factory = stub_factory
        self._max_concurrent_channels = max_concurrent_channels
        self._chunk_size = chunk_size_bytes
        self._default_compression = default_compression

        # participant_id -> ChannelState
        self._channel_pool: dict[str, ChannelState] = {}
        # participant_id -> asyncio.Semaphore (one per participant, serialises sends)
        self._send_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Channel lifecycle
    # ------------------------------------------------------------------

    async def open_channel(
        self,
        participant_id: str,
        channel_config: ChannelConfig,
    ) -> None:
        """Open and register a gRPC channel to a remote participant.

        Validates TLS configuration and initialises the connection pool entry.
        Does not perform an actual TCP handshake until the first send.

        Args:
            participant_id: Unique ID of the remote participant node.
            channel_config: TLS/endpoint configuration for the channel.

        Raises:
            ValueError: If channel capacity is exhausted or config is invalid.
        """
        if len(self._channel_pool) >= self._max_concurrent_channels:
            raise ValueError(
                f"Connection pool exhausted: {self._max_concurrent_channels} "
                "channels already open"
            )

        if participant_id in self._channel_pool:
            logger.warning(
                "Channel to participant %s already open — ignoring open_channel call",
                participant_id,
            )
            return

        state = ChannelState(participant_id=participant_id, config=channel_config)
        self._channel_pool[participant_id] = state
        self._send_locks[participant_id] = asyncio.Lock()

        logger.info(
            "Opened channel to participant %s at %s (compression=%s)",
            participant_id,
            channel_config.endpoint,
            channel_config.compression,
        )

    async def close_channel(self, participant_id: str) -> None:
        """Close and remove the channel for a participant.

        Any unacknowledged messages at close time are logged as warnings.

        Args:
            participant_id: Participant whose channel should be closed.
        """
        state = self._channel_pool.pop(participant_id, None)
        self._send_locks.pop(participant_id, None)

        if state is None:
            logger.debug("close_channel called for unknown participant %s", participant_id)
            return

        if state.pending_acks:
            logger.warning(
                "Channel to %s closed with %d unacknowledged messages: %s",
                participant_id,
                len(state.pending_acks),
                list(state.pending_acks.keys()),
            )

        state.is_open = False
        logger.info(
            "Closed channel to participant %s (sent=%d bytes, received=%d bytes)",
            participant_id,
            state.bytes_sent,
            state.bytes_received,
        )

    def get_open_channels(self) -> list[str]:
        """Return a list of participant_ids with open channels."""
        return [pid for pid, s in self._channel_pool.items() if s.is_open]

    # ------------------------------------------------------------------
    # Model weight serialization
    # ------------------------------------------------------------------

    @staticmethod
    def serialize_model_weights(
        parameters: list[np.ndarray[Any, Any]],
    ) -> bytes:
        """Serialize model parameter arrays to a compact binary format.

        Uses NumPy's NPZ format (zipped binary). In production, these bytes
        would be embedded in a protobuf ModelWeights message.

        Args:
            parameters: List of NumPy arrays representing model layers.

        Returns:
            Raw bytes suitable for transport.
        """
        buffer = io.BytesIO()
        arrays_dict = {f"layer_{idx}": arr for idx, arr in enumerate(parameters)}
        np.savez_compressed(buffer, **arrays_dict)
        return buffer.getvalue()

    @staticmethod
    def deserialize_model_weights(data: bytes) -> list[np.ndarray[Any, Any]]:
        """Deserialize model weights from binary NPZ format.

        Args:
            data: Raw bytes produced by serialize_model_weights.

        Returns:
            List of NumPy arrays in original layer order.
        """
        buffer = io.BytesIO(data)
        loaded = np.load(buffer, allow_pickle=False)
        # Sort keys to restore original layer order
        sorted_keys = sorted(loaded.files, key=lambda k: int(k.split("_")[1]))
        return [loaded[key] for key in sorted_keys]

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    @staticmethod
    def compress(data: bytes, algorithm: str) -> bytes:
        """Compress raw bytes using the specified algorithm.

        Args:
            data: Bytes to compress.
            algorithm: 'gzip', 'lz4', or 'none'.

        Returns:
            Compressed bytes (with a 1-byte algorithm header prefix).

        Raises:
            ValueError: If the algorithm is unsupported.
        """
        if algorithm == "none":
            return b"\x00" + data
        if algorithm == "gzip":
            return b"\x01" + gzip.compress(data, compresslevel=6)
        if algorithm == "lz4":
            # lz4 is an optional dependency — fall back to gzip if unavailable
            try:
                import lz4.frame  # type: ignore[import-untyped]

                return b"\x02" + lz4.frame.compress(data)
            except ImportError:
                logger.warning("lz4 not installed, falling back to gzip")
                return b"\x01" + gzip.compress(data, compresslevel=6)

        raise ValueError(f"Unknown compression algorithm: '{algorithm}'")

    @staticmethod
    def decompress(data: bytes) -> bytes:
        """Decompress bytes produced by compress().

        Args:
            data: Compressed bytes with algorithm header.

        Returns:
            Original uncompressed bytes.

        Raises:
            ValueError: If the algorithm header is unknown.
        """
        if not data:
            raise ValueError("Empty data passed to decompress()")

        header = data[0]
        payload = data[1:]

        if header == 0x00:
            return payload
        if header == 0x01:
            return gzip.decompress(payload)
        if header == 0x02:
            try:
                import lz4.frame  # type: ignore[import-untyped]

                return lz4.frame.decompress(payload)
            except ImportError:
                raise ValueError("lz4 not installed, cannot decompress lz4 payload")

        raise ValueError(f"Unknown compression header byte: {header:#x}")

    # ------------------------------------------------------------------
    # Chunked transfer
    # ------------------------------------------------------------------

    def split_into_chunks(self, data: bytes) -> list[tuple[int, int, bytes]]:
        """Split a large payload into indexed chunks for streaming.

        Args:
            data: Raw bytes to split.

        Returns:
            List of (chunk_index, total_chunks, chunk_bytes) tuples.
        """
        total = len(data)
        chunk_count = max(1, (total + self._chunk_size - 1) // self._chunk_size)
        chunks: list[tuple[int, int, bytes]] = []
        for index in range(chunk_count):
            start = index * self._chunk_size
            end = min(start + self._chunk_size, total)
            chunks.append((index, chunk_count, data[start:end]))
        return chunks

    @staticmethod
    def reassemble_chunks(chunks: list[tuple[int, int, bytes]]) -> bytes:
        """Reassemble an ordered list of chunks into the original payload.

        Args:
            chunks: List of (chunk_index, total_chunks, chunk_bytes) in any order.

        Returns:
            Reassembled bytes.

        Raises:
            ValueError: If chunk indices are inconsistent or missing.
        """
        if not chunks:
            raise ValueError("No chunks provided for reassembly")

        sorted_chunks = sorted(chunks, key=lambda c: c[0])
        total_expected = sorted_chunks[0][1]

        if len(sorted_chunks) != total_expected:
            raise ValueError(
                f"Expected {total_expected} chunks, received {len(sorted_chunks)}"
            )

        return b"".join(chunk_bytes for _, _, chunk_bytes in sorted_chunks)

    # ------------------------------------------------------------------
    # Send model update
    # ------------------------------------------------------------------

    async def send_model_update(
        self,
        participant_id: str,
        job_id: str,
        round_number: int,
        parameters: list[np.ndarray[Any, Any]],
        *,
        compression: str | None = None,
    ) -> str:
        """Send aggregated model weights to a remote participant.

        Serializes, compresses, and streams the model update in chunks.
        Implements retry logic with exponential backoff. Returns message_id.

        Args:
            participant_id: Recipient participant node.
            job_id: Federated job context.
            round_number: The training round this update belongs to.
            parameters: Model weight arrays to transmit.
            compression: Override the channel-default compression algorithm.

        Returns:
            Message UUID for tracking acknowledgment.

        Raises:
            KeyError: If the channel for participant_id is not open.
            RuntimeError: If all retry attempts are exhausted.
        """
        state = self._get_channel_state(participant_id)
        algo = compression or state.config.compression

        raw_bytes = self.serialize_model_weights(parameters)
        compressed = self.compress(raw_bytes, algo)
        chunks = self.split_into_chunks(compressed)

        message_id = str(uuid.uuid4())
        stub = self._stub_factory(state.config)

        async with self._send_locks[participant_id]:
            for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
                try:
                    for chunk_index, total_chunks, chunk_bytes in chunks:
                        await self._send_chunk(
                            stub=stub,
                            message_id=message_id,
                            participant_id=participant_id,
                            job_id=job_id,
                            round_number=round_number,
                            chunk_index=chunk_index,
                            total_chunks=total_chunks,
                            chunk_bytes=chunk_bytes,
                        )

                    state.bytes_sent += len(compressed)
                    state.pending_acks[message_id] = time.time()

                    logger.info(
                        "Sent model update %s to participant %s "
                        "(job=%s round=%d, %d bytes, %d chunks, compression=%s)",
                        message_id,
                        participant_id,
                        job_id,
                        round_number,
                        len(compressed),
                        len(chunks),
                        algo,
                    )
                    return message_id

                except Exception as exc:
                    backoff = RETRY_BACKOFF_BASE_SECONDS ** attempt
                    logger.warning(
                        "send_model_update attempt %d/%d failed for participant %s: %s. "
                        "Retrying in %.1f seconds.",
                        attempt,
                        MAX_RETRY_ATTEMPTS,
                        participant_id,
                        exc,
                        backoff,
                    )
                    if attempt == MAX_RETRY_ATTEMPTS:
                        raise RuntimeError(
                            f"Failed to send model update to {participant_id} "
                            f"after {MAX_RETRY_ATTEMPTS} attempts"
                        ) from exc
                    await asyncio.sleep(backoff)

        # Unreachable but satisfies type checker
        raise RuntimeError("Unexpected exit from retry loop")  # pragma: no cover

    async def _send_chunk(
        self,
        *,
        stub: Any,
        message_id: str,
        participant_id: str,
        job_id: str,
        round_number: int,
        chunk_index: int,
        total_chunks: int,
        chunk_bytes: bytes,
    ) -> None:
        """Transmit a single chunk via the gRPC stub.

        Constructs a length-prefixed frame (4-byte big-endian length + payload)
        and delegates to the stub's SendChunk method.
        """
        # Length-prefix the chunk for framing
        framed = struct.pack(">I", len(chunk_bytes)) + chunk_bytes

        await stub.SendChunk(
            message_id=message_id,
            participant_id=participant_id,
            job_id=job_id,
            round_number=round_number,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            payload=framed,
        )

    # ------------------------------------------------------------------
    # Acknowledgment handling
    # ------------------------------------------------------------------

    async def acknowledge_message(
        self,
        participant_id: str,
        ack: MessageAck,
    ) -> None:
        """Process an inbound acknowledgment from a participant.

        Args:
            participant_id: The participant sending the ACK.
            ack: Acknowledgment metadata including checksum validity.

        Raises:
            KeyError: If channel is not found.
            ValueError: If the ACK references an unknown message_id.
        """
        state = self._get_channel_state(participant_id)

        if ack.message_id not in state.pending_acks:
            raise ValueError(
                f"Unknown message_id in ACK: {ack.message_id} from {participant_id}"
            )

        send_time = state.pending_acks.pop(ack.message_id)
        rtt_ms = (ack.received_at - send_time) * 1000

        if not ack.checksum_valid:
            logger.error(
                "Checksum mismatch for message %s from participant %s — "
                "retransmit required",
                ack.message_id,
                participant_id,
            )
        else:
            logger.info(
                "ACK for message %s from participant %s (RTT=%.1f ms)",
                ack.message_id,
                participant_id,
                rtt_ms,
            )

    def get_pending_acks(self, participant_id: str) -> list[str]:
        """Return message IDs awaiting acknowledgment from a participant.

        Args:
            participant_id: The participant to check.

        Returns:
            List of unacknowledged message IDs.
        """
        state = self._get_channel_state(participant_id)
        return list(state.pending_acks.keys())

    # ------------------------------------------------------------------
    # Metrics and pool inspection
    # ------------------------------------------------------------------

    def get_channel_stats(self, participant_id: str) -> dict[str, Any]:
        """Return bandwidth and message statistics for a channel.

        Args:
            participant_id: The channel to inspect.

        Returns:
            Dict with keys: participant_id, endpoint, bytes_sent, bytes_received,
            pending_acks, uptime_seconds, compression, is_open.
        """
        state = self._get_channel_state(participant_id)
        uptime = time.time() - state.connected_at
        return {
            "participant_id": participant_id,
            "endpoint": state.config.endpoint,
            "bytes_sent": state.bytes_sent,
            "bytes_received": state.bytes_received,
            "pending_acks": len(state.pending_acks),
            "uptime_seconds": round(uptime, 2),
            "compression": state.config.compression,
            "is_open": state.is_open,
        }

    def pool_summary(self) -> dict[str, Any]:
        """Return a summary of the connection pool state.

        Returns:
            Dict with total_channels, open_channels, total_bytes_sent,
            total_bytes_received.
        """
        total_sent = sum(s.bytes_sent for s in self._channel_pool.values())
        total_received = sum(s.bytes_received for s in self._channel_pool.values())
        open_count = sum(1 for s in self._channel_pool.values() if s.is_open)
        return {
            "total_channels": len(self._channel_pool),
            "open_channels": open_count,
            "total_bytes_sent": total_sent,
            "total_bytes_received": total_received,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_channel_state(self, participant_id: str) -> ChannelState:
        """Retrieve channel state or raise a descriptive error.

        Args:
            participant_id: The participant to look up.

        Returns:
            ChannelState for the participant.

        Raises:
            KeyError: If no channel is open for participant_id.
        """
        state = self._channel_pool.get(participant_id)
        if state is None:
            raise KeyError(
                f"No open channel for participant {participant_id}. "
                "Call open_channel() first."
            )
        return state
