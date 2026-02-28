"""AumOS FL participant authentication SDK.

Gap #147: Client SDK for Participants — authentication layer.

Provides the ``join_fl_job()`` function that organisations call before starting
their Flower client. Returns the Flower server address and TLS certificate.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import httpx

from aumos_common.observability import get_logger

logger = get_logger(__name__)


@dataclass
class ParticipantCredentials:
    """Credentials returned when an organisation joins an FL job.

    Attributes:
        participant_id: UUID assigned to this participant.
        flower_server_address: Flower gRPC server address (host:port).
        tls_cert_pem: PEM-encoded TLS certificate for the Flower server.
        job_id: The FL job this participant has been enrolled in.
    """

    participant_id: str
    flower_server_address: str
    tls_cert_pem: str | None
    job_id: str


async def join_fl_job(
    aumos_api_url: str,
    job_id: str,
    organization_api_key: str,
    organization_name: str,
    data_size: int | None = None,
) -> ParticipantCredentials:
    """Authenticate with the AumOS FL server and obtain participant credentials.

    This must be called before starting the Flower client. The returned
    credentials include the Flower server address and optional TLS certificate.

    Args:
        aumos_api_url: Base URL of the AumOS FL API (e.g. ``https://api.aumos.ai``).
        job_id: The FL job UUID to join.
        organization_api_key: The organisation's AumOS API key.
        organization_name: Human-readable name for this participant.
        data_size: Optional size of the local training dataset.

    Returns:
        ParticipantCredentials including ``flower_server_address`` and ``tls_cert_pem``.

    Raises:
        httpx.HTTPStatusError: If the server rejects the join request.
    """
    payload: dict[str, object] = {
        "job_id": job_id,
        "organization_name": organization_name,
    }
    if data_size is not None:
        payload["data_size"] = data_size

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{aumos_api_url}/fl/jobs/{job_id}/join",
            json=payload,
            headers={"X-API-Key": organization_api_key},
        )
        response.raise_for_status()
        data = response.json()

    credentials = ParticipantCredentials(
        participant_id=data.get("participant_id", str(uuid.uuid4())),
        flower_server_address=data.get("flower_server_address", "localhost:8080"),
        tls_cert_pem=data.get("tls_cert_pem"),
        job_id=job_id,
    )
    logger.info(
        "Joined FL job",
        job_id=job_id,
        participant_id=credentials.participant_id,
        flower_server_address=credentials.flower_server_address,
    )
    return credentials


async def get_global_model(
    aumos_api_url: str,
    job_id: str,
    organization_api_key: str,
    output_path: str,
) -> str:
    """Download the current global model checkpoint for a completed FL job.

    Args:
        aumos_api_url: Base URL of the AumOS FL API.
        job_id: The FL job UUID.
        organization_api_key: The organisation's AumOS API key.
        output_path: Local path to save the model file.

    Returns:
        The URI of the downloaded model artifact.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.get(
            f"{aumos_api_url}/fl/jobs/{job_id}/model",
            headers={"X-API-Key": organization_api_key},
        )
        response.raise_for_status()
        data = response.json()
        model_uri: str = data.get("aggregated_model_uri", "")

    logger.info("Global model downloaded", job_id=job_id, output_path=output_path, uri=model_uri)
    return model_uri
