"""Pydantic v2 request and response schemas for the FL API."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Job schemas
# ---------------------------------------------------------------------------


class CreateJobRequest(BaseModel):
    """Request body for POST /fl/jobs."""

    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=2048)
    strategy: str = Field(default="fedavg", pattern="^(fedavg|fedprox|scaffold|custom)$")
    num_rounds: int = Field(default=10, ge=1, le=1000)
    min_participants: int = Field(default=2, ge=1, le=1000)

    # Differential privacy — omit to disable DP
    dp_epsilon: float | None = Field(default=None, gt=0.0, description="Privacy budget ε")
    dp_delta: float | None = Field(
        default=None, gt=0.0, lt=1.0, description="Privacy failure probability δ"
    )

    # Strategy-specific config (e.g. proximal_mu for FedProx)
    strategy_config: dict[str, Any] | None = Field(default=None)


class JobResponse(BaseModel):
    """Response schema for a federated learning job."""

    id: uuid.UUID
    tenant_id: str
    name: str
    description: str | None
    status: str
    strategy: str
    num_rounds: int
    current_round: int
    min_participants: int
    actual_participants: int
    dp_epsilon: Decimal | None
    dp_delta: Decimal | None
    aggregated_model_uri: str | None
    synthetic_fallback_used: bool
    strategy_config: dict[str, Any] | None

    model_config = {"from_attributes": True}


class StartJobRequest(BaseModel):
    """Optional configuration for POST /fl/jobs/{id}/start."""

    data_schema: dict[str, Any] | None = Field(
        default=None,
        description="Schema of training data for synthetic fallback if needed",
    )
    allow_synthetic_fallback: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Participant schemas
# ---------------------------------------------------------------------------


class JoinJobRequest(BaseModel):
    """Request body for POST /fl/participants."""

    job_id: uuid.UUID
    organization_name: str = Field(min_length=1, max_length=256)
    organization_id: str | None = Field(default=None, max_length=128)
    data_size: int | None = Field(default=None, ge=1)
    data_description: str | None = Field(default=None, max_length=2048)

    # RSA/EC public key PEM for secure aggregation key exchange
    public_key_pem: str | None = Field(default=None)


class ParticipantResponse(BaseModel):
    """Response schema for a federated learning participant."""

    id: uuid.UUID
    job_id: uuid.UUID
    tenant_id: str
    organization_name: str
    organization_id: str | None
    status: str
    data_size: int | None
    contribution_weight: Decimal | None
    rounds_completed: int

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Round schemas
# ---------------------------------------------------------------------------


class SubmitUpdateRequest(BaseModel):
    """Request body for POST /fl/jobs/{id}/rounds/{num}/submit."""

    participant_id: uuid.UUID
    update_uri: str = Field(
        min_length=1, description="URI pointing to the participant's model update artifact"
    )
    num_samples: int = Field(ge=1, description="Number of training samples used")
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Training metrics (loss, accuracy, etc.)"
    )


class RoundResponse(BaseModel):
    """Response schema for an aggregation round."""

    id: uuid.UUID
    job_id: uuid.UUID
    round_number: int
    participants_submitted: int
    aggregation_method: str
    dp_noise_added: bool
    round_metrics: dict[str, Any] | None
    round_model_uri: str | None
    started_at: str | None
    completed_at: str | None

    model_config = {"from_attributes": True}


class RoundListResponse(BaseModel):
    """Paginated list of aggregation rounds."""

    rounds: list[RoundResponse]
    total: int


# ---------------------------------------------------------------------------
# Model download schema
# ---------------------------------------------------------------------------


class ModelDownloadResponse(BaseModel):
    """Response for GET /fl/jobs/{id}/model."""

    job_id: uuid.UUID
    aggregated_model_uri: str
    current_round: int
    num_rounds: int
    strategy: str
    dp_epsilon: Decimal | None
    dp_delta: Decimal | None


# ---------------------------------------------------------------------------
# Generic response schemas
# ---------------------------------------------------------------------------


class MessageResponse(BaseModel):
    """Simple message response for acknowledgement endpoints."""

    message: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None


# ---------------------------------------------------------------------------
# Simulation schemas (Gap #146)
# ---------------------------------------------------------------------------


class SimulationRequest(BaseModel):
    """Request body for POST /fl/simulate."""

    strategy: str = Field(default="fedavg", pattern="^(fedavg|fedprox|scaffold)$")
    num_clients: int = Field(default=10, ge=2, le=1000)
    num_rounds: int = Field(default=10, ge=1, le=200)
    fraction_fit: float = Field(default=0.5, gt=0.0, le=1.0)
    dp_epsilon: float | None = Field(default=None, gt=0.0)
    fedprox_mu: float = Field(default=0.01, ge=0.0)


class SimulationRoundMetrics(BaseModel):
    """Per-round metrics from a simulation."""

    round_number: int
    distributed_loss: float
    centralized_accuracy: float


class SimulationResponse(BaseModel):
    """Response schema for a completed FL simulation."""

    simulation_id: str
    strategy: str
    num_rounds: int
    num_clients: int
    per_round_metrics: list[SimulationRoundMetrics]
    final_accuracy: float


# ---------------------------------------------------------------------------
# Async update submission schemas (Gap #149)
# ---------------------------------------------------------------------------


class UpdateSubmissionRequest(BaseModel):
    """Request body for POST /fl/jobs/{id}/updates (async update submission)."""

    participant_id: uuid.UUID
    client_round: int = Field(ge=0, description="Round when client started training")
    update_uri: str = Field(min_length=1)
    num_samples: int = Field(ge=1)
    metrics: dict[str, Any] = Field(default_factory=dict)


class GlobalWeightsResponse(BaseModel):
    """Response schema for GET /fl/jobs/{id}/global-weights."""

    job_id: uuid.UUID
    current_round: int
    weights_uri: str
    strategy: str
    num_participants_aggregated: int


# ---------------------------------------------------------------------------
# Participant credentials (Gap #147)
# ---------------------------------------------------------------------------


class ParticipantCredentials(BaseModel):
    """Credentials returned when an organization joins an FL job."""

    participant_id: str
    flower_server_address: str
    tls_cert_pem: str | None = None
    job_id: uuid.UUID


# ---------------------------------------------------------------------------
# Federated analytics schemas (Gap #152)
# ---------------------------------------------------------------------------


class AnalyticsQueryRequest(BaseModel):
    """Request body for POST /fl/analytics/query."""

    aggregation_type: str = Field(
        pattern="^(count|sum|mean|variance|histogram)$",
        description="Type of aggregate to compute",
    )
    column_name: str | None = Field(default=None, max_length=128)
    bins: int | None = Field(default=10, ge=2, le=1000)
    range_min: float | None = Field(default=0.0)
    range_max: float | None = Field(default=1.0)
    epsilon: float = Field(default=1.0, gt=0.0, description="DP budget epsilon for this query")


class AnalyticsResultResponse(BaseModel):
    """Response schema for a completed federated analytics query."""

    aggregation_type: str
    result: Any
    total_count: int
    num_participants: int
    epsilon_consumed: float
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# TEE attestation schemas (Gap #153)
# ---------------------------------------------------------------------------


class AttestationNonceResponse(BaseModel):
    """Response for GET /fl/attestation/nonce — issues anti-replay nonce."""

    nonce: str
    participant_id: str
    job_id: uuid.UUID


class AttestationQuoteRequest(BaseModel):
    """Request body for POST /fl/attestation/verify."""

    participant_id: str
    job_id: uuid.UUID
    raw_quote_b64: str = Field(
        min_length=1, description="Base64-encoded Intel SGX QUOTE_t"
    )


class AttestationQuoteResponse(BaseModel):
    """Response schema after SGX quote verification."""

    participant_id: str
    job_id: uuid.UUID
    mrenclave: str
    verified: bool
    verification_metadata: dict[str, Any] = Field(default_factory=dict)
