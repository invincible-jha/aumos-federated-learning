"""FastAPI router for federated learning job management.

Endpoints:
    POST   /fl/jobs                          — create FL job
    GET    /fl/jobs/{id}                     — job status and progress
    POST   /fl/jobs/{id}/start               — start training
    POST   /fl/participants                  — join as participant
    GET    /fl/jobs/{id}/rounds              — round history
    POST   /fl/jobs/{id}/rounds/{num}/submit — submit model update
    GET    /fl/jobs/{id}/model               — download aggregated model
    POST   /fl/simulate                      — run in-process simulation (Gap #146)
    POST   /fl/jobs/{id}/updates             — async update submission (Gap #149)
    GET    /fl/jobs/{id}/global-weights      — retrieve current global model (Gap #149)
    POST   /fl/analytics/query               — federated analytics query (Gap #152)
    GET    /fl/attestation/nonce             — issue SGX attestation nonce (Gap #153)
    POST   /fl/attestation/verify            — verify SGX quote (Gap #153)
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from aumos_federated_learning.api.schemas import (
    AnalyticsQueryRequest,
    AnalyticsResultResponse,
    AttestationNonceResponse,
    AttestationQuoteRequest,
    AttestationQuoteResponse,
    CreateJobRequest,
    GlobalWeightsResponse,
    JobResponse,
    JoinJobRequest,
    MessageResponse,
    ModelDownloadResponse,
    ParticipantCredentials,
    ParticipantResponse,
    RoundListResponse,
    RoundResponse,
    SimulationRequest,
    SimulationResponse,
    SimulationRoundMetrics,
    StartJobRequest,
    SubmitUpdateRequest,
    UpdateSubmissionRequest,
)

router = APIRouter(prefix="/fl", tags=["federated-learning"])


# ---------------------------------------------------------------------------
# Stub dependency providers
# In production these are wired via FastAPI dependency injection from adapters.
# ---------------------------------------------------------------------------


def _get_job_service() -> Any:
    """Provide JobService — override in production with proper DI."""
    raise NotImplementedError("JobService not wired — configure DI in main.py")


def _get_coordination_service() -> Any:
    """Provide CoordinationService."""
    raise NotImplementedError("CoordinationService not wired")


def _get_training_service() -> Any:
    """Provide TrainingService."""
    raise NotImplementedError("TrainingService not wired")


def _get_simulation_service() -> Any:
    """Provide SimulationService — override in production with proper DI."""
    raise NotImplementedError("SimulationService not wired")


def _get_analytics_service() -> Any:
    """Provide FederatedAnalyticsService — override in production with proper DI."""
    raise NotImplementedError("FederatedAnalyticsService not wired")


def _get_attestation_service() -> Any:
    """Provide TEEAttestationService — override in production with proper DI."""
    raise NotImplementedError("TEEAttestationService not wired")


def _get_current_tenant() -> str:
    """Extract tenant_id from request context — override with auth middleware."""
    return "default-tenant"


JobServiceDep = Annotated[Any, Depends(_get_job_service)]
CoordinationServiceDep = Annotated[Any, Depends(_get_coordination_service)]
TrainingServiceDep = Annotated[Any, Depends(_get_training_service)]
SimulationServiceDep = Annotated[Any, Depends(_get_simulation_service)]
AnalyticsServiceDep = Annotated[Any, Depends(_get_analytics_service)]
AttestationServiceDep = Annotated[Any, Depends(_get_attestation_service)]
TenantDep = Annotated[str, Depends(_get_current_tenant)]


# ---------------------------------------------------------------------------
# Job endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/jobs",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a federated learning job",
)
async def create_job(
    body: CreateJobRequest,
    job_service: JobServiceDep,
    tenant_id: TenantDep,
) -> Any:
    """Create a new federated learning job in 'configuring' status.

    The job will remain in 'configuring' until explicitly started via POST /fl/jobs/{id}/start.
    """
    job = await job_service.create_job(
        tenant_id=tenant_id,
        name=body.name,
        description=body.description,
        strategy=body.strategy,
        num_rounds=body.num_rounds,
        min_participants=body.min_participants,
        dp_epsilon=body.dp_epsilon,
        dp_delta=body.dp_delta,
        strategy_config=body.strategy_config,
    )
    return job


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    summary="Get job status and progress",
)
async def get_job(
    job_id: Annotated[uuid.UUID, Path(description="Federated job ID")],
    job_service: JobServiceDep,
    tenant_id: TenantDep,
) -> Any:
    """Retrieve current status and progress of a federated learning job."""
    job = await job_service.get_job(job_id=job_id, tenant_id=tenant_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


@router.post(
    "/jobs/{job_id}/start",
    response_model=MessageResponse,
    summary="Start a federated learning job",
)
async def start_job(
    job_id: Annotated[uuid.UUID, Path(description="Federated job ID")],
    body: StartJobRequest,
    job_service: JobServiceDep,
    tenant_id: TenantDep,
) -> MessageResponse:
    """Start a configured job.

    Transitions status: configuring → recruiting.
    If allow_synthetic_fallback is True and actual participants < min_participants,
    synthetic data fallback will be triggered automatically.
    """
    try:
        await job_service.start_job(job_id=job_id, tenant_id=tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return MessageResponse(message=f"Job {job_id} started — now recruiting participants")


# ---------------------------------------------------------------------------
# Participant endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/participants",
    response_model=ParticipantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Join a federated learning job as a participant",
)
async def join_job(
    body: JoinJobRequest,
    coordination_service: CoordinationServiceDep,
    tenant_id: TenantDep,
) -> Any:
    """Register an organization as a participant in a federated learning job.

    The participant starts in 'invited' status and must accept before training begins.
    Public key PEM is used for MPC-based secure aggregation key exchange.
    """
    try:
        participant = await coordination_service.add_participant(
            job_id=body.job_id,
            tenant_id=tenant_id,
            organization_name=body.organization_name,
            organization_id=body.organization_id,
            data_size=body.data_size,
            public_key_pem=body.public_key_pem,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return participant


# ---------------------------------------------------------------------------
# Round endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/jobs/{job_id}/rounds",
    response_model=RoundListResponse,
    summary="List round history for a job",
)
async def list_rounds(
    job_id: Annotated[uuid.UUID, Path(description="Federated job ID")],
    training_service: TrainingServiceDep,
    tenant_id: TenantDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> RoundListResponse:
    """Retrieve the history of aggregation rounds for a job."""
    rounds = await training_service.get_round_history(job_id=job_id, tenant_id=tenant_id)
    paginated = rounds[offset : offset + limit]
    return RoundListResponse(
        rounds=[RoundResponse.model_validate(r) for r in paginated],
        total=len(rounds),
    )


@router.post(
    "/jobs/{job_id}/rounds/{round_number}/submit",
    response_model=MessageResponse,
    summary="Submit a model update for a training round",
)
async def submit_update(
    job_id: Annotated[uuid.UUID, Path(description="Federated job ID")],
    round_number: Annotated[int, Path(ge=1, description="Round number")],
    body: SubmitUpdateRequest,
    training_service: TrainingServiceDep,
    tenant_id: TenantDep,
) -> MessageResponse:
    """Submit a model update artifact for a specific training round.

    The update_uri should point to the participant's encrypted gradient/weight update.
    Raw training data is never transmitted — only model update artifacts.
    """
    try:
        await training_service.submit_update(
            job_id=job_id,
            round_number=round_number,
            participant_id=body.participant_id,
            update_uri=body.update_uri,
            num_samples=body.num_samples,
            metrics=body.metrics,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return MessageResponse(
        message=f"Update submitted for job {job_id} round {round_number}"
    )


# ---------------------------------------------------------------------------
# Model download endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/jobs/{job_id}/model",
    response_model=ModelDownloadResponse,
    summary="Get the aggregated model for a completed job",
)
async def get_model(
    job_id: Annotated[uuid.UUID, Path(description="Federated job ID")],
    job_service: JobServiceDep,
    tenant_id: TenantDep,
) -> ModelDownloadResponse:
    """Retrieve the final aggregated model URI for a completed federated learning job."""
    job = await job_service.get_job(job_id=job_id, tenant_id=tenant_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if job.aggregated_model_uri is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is not complete — current status: {job.status}",
        )
    return ModelDownloadResponse(
        job_id=job.id,
        aggregated_model_uri=job.aggregated_model_uri,
        current_round=job.current_round,
        num_rounds=job.num_rounds,
        strategy=job.strategy,
        dp_epsilon=job.dp_epsilon,
        dp_delta=job.dp_delta,
    )


# ---------------------------------------------------------------------------
# Simulation endpoint (Gap #146)
# ---------------------------------------------------------------------------


@router.post(
    "/simulate",
    response_model=SimulationResponse,
    status_code=status.HTTP_200_OK,
    summary="Run an in-process FL simulation",
)
async def run_simulation(
    body: SimulationRequest,
    simulation_service: SimulationServiceDep,
    tenant_id: TenantDep,
) -> SimulationResponse:
    """Run a complete FL simulation in-process using Flower's simulation module.

    No real participant connections or network setup required. Returns per-round
    convergence metrics for strategy comparison.
    """
    try:
        result = await simulation_service.run_simulation(
            tenant_id=tenant_id,
            strategy=body.strategy,
            num_clients=body.num_clients,
            num_rounds=body.num_rounds,
            fraction_fit=body.fraction_fit,
            dp_epsilon=body.dp_epsilon,
            fedprox_mu=body.fedprox_mu,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Simulation dependencies not installed: {exc}",
        ) from exc

    return SimulationResponse(
        simulation_id=result.simulation_id,
        strategy=result.strategy,
        num_rounds=result.num_rounds,
        num_clients=result.num_clients,
        per_round_metrics=[
            SimulationRoundMetrics(
                round_number=m.round_number,
                distributed_loss=m.distributed_loss,
                centralized_accuracy=m.centralized_accuracy,
            )
            for m in result.per_round_metrics
        ],
        final_accuracy=result.final_accuracy,
    )


# ---------------------------------------------------------------------------
# Async update submission and global weights (Gap #149)
# ---------------------------------------------------------------------------


@router.post(
    "/jobs/{job_id}/updates",
    response_model=MessageResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an asynchronous model update",
)
async def submit_async_update(
    job_id: Annotated[uuid.UUID, Path(description="Federated job ID")],
    body: UpdateSubmissionRequest,
    training_service: TrainingServiceDep,
    tenant_id: TenantDep,
) -> MessageResponse:
    """Submit a model update asynchronously, without waiting for all participants.

    The server applies staleness-aware weighting (FedAsync) and triggers
    aggregation once the buffer is full.
    """
    try:
        await training_service.submit_async_update(
            job_id=job_id,
            tenant_id=tenant_id,
            participant_id=body.participant_id,
            client_round=body.client_round,
            update_uri=body.update_uri,
            num_samples=body.num_samples,
            metrics=body.metrics,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return MessageResponse(message=f"Async update accepted for job {job_id}")


@router.get(
    "/jobs/{job_id}/global-weights",
    response_model=GlobalWeightsResponse,
    summary="Retrieve current global model weights URI",
)
async def get_global_weights(
    job_id: Annotated[uuid.UUID, Path(description="Federated job ID")],
    job_service: JobServiceDep,
    tenant_id: TenantDep,
) -> GlobalWeightsResponse:
    """Return the current global model weights URI for asynchronous clients."""
    job = await job_service.get_job(job_id=job_id, tenant_id=tenant_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if job.aggregated_model_uri is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No global model available yet — waiting for first aggregation",
        )
    return GlobalWeightsResponse(
        job_id=job.id,
        current_round=job.current_round,
        weights_uri=job.aggregated_model_uri,
        strategy=job.strategy,
        num_participants_aggregated=job.actual_participants,
    )


# ---------------------------------------------------------------------------
# Federated analytics endpoint (Gap #152)
# ---------------------------------------------------------------------------


@router.post(
    "/analytics/query",
    response_model=AnalyticsResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a federated analytics query with differential privacy",
)
async def run_analytics_query(
    body: AnalyticsQueryRequest,
    analytics_service: AnalyticsServiceDep,
    tenant_id: TenantDep,
) -> AnalyticsResultResponse:
    """Compute a DP-protected aggregate statistic across FL participants.

    Supported aggregation types: count, sum, mean, variance, histogram.
    All results are protected by the Laplace mechanism with the specified epsilon.
    """
    try:
        result = await analytics_service.run_query(
            tenant_id=tenant_id,
            aggregation_type=body.aggregation_type,
            column_name=body.column_name,
            bins=body.bins,
            range_min=body.range_min,
            range_max=body.range_max,
            epsilon=body.epsilon,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    return AnalyticsResultResponse(
        aggregation_type=result.aggregation_type.value,
        result=result.result,
        total_count=result.total_count,
        num_participants=result.num_participants,
        epsilon_consumed=result.epsilon_consumed,
        metadata=result.metadata,
    )


# ---------------------------------------------------------------------------
# TEE attestation endpoints (Gap #153)
# ---------------------------------------------------------------------------


@router.get(
    "/attestation/nonce",
    response_model=AttestationNonceResponse,
    summary="Issue an anti-replay nonce for SGX attestation",
)
async def get_attestation_nonce(
    job_id: Annotated[uuid.UUID, Query(description="FL job requiring attestation")],
    participant_id: Annotated[str, Query(min_length=1)],
    attestation_service: AttestationServiceDep,
    tenant_id: TenantDep,
) -> AttestationNonceResponse:
    """Issue a fresh nonce that the participant must embed in their SGX report data."""
    nonce = await attestation_service.issue_nonce(
        job_id=str(job_id),
        participant_id=participant_id,
        tenant_id=tenant_id,
    )
    return AttestationNonceResponse(
        nonce=nonce,
        participant_id=participant_id,
        job_id=job_id,
    )


@router.post(
    "/attestation/verify",
    response_model=AttestationQuoteResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify an Intel SGX attestation quote",
)
async def verify_attestation_quote(
    body: AttestationQuoteRequest,
    attestation_service: AttestationServiceDep,
    tenant_id: TenantDep,
) -> AttestationQuoteResponse:
    """Verify a participant's SGX attestation quote.

    On success, the participant is added to the attested allowlist for the job.
    Only attested participants may receive global model weights in TEE-enforced jobs.
    """
    try:
        quote = await attestation_service.process_quote(
            job_id=str(body.job_id),
            participant_id=body.participant_id,
            raw_quote_b64=body.raw_quote_b64,
            tenant_id=tenant_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Attestation verification failed: {exc}",
        ) from exc

    return AttestationQuoteResponse(
        participant_id=body.participant_id,
        job_id=body.job_id,
        mrenclave=quote.mrenclave,
        verified=quote.verified,
        verification_metadata=quote.verification_metadata,
    )
