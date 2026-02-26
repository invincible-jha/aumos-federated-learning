"""FastAPI router for federated learning job management.

Endpoints:
    POST   /fl/jobs                          — create FL job
    GET    /fl/jobs/{id}                     — job status and progress
    POST   /fl/jobs/{id}/start               — start training
    POST   /fl/participants                  — join as participant
    GET    /fl/jobs/{id}/rounds              — round history
    POST   /fl/jobs/{id}/rounds/{num}/submit — submit model update
    GET    /fl/jobs/{id}/model               — download aggregated model
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from aumos_federated_learning.api.schemas import (
    CreateJobRequest,
    JobResponse,
    JoinJobRequest,
    MessageResponse,
    ModelDownloadResponse,
    ParticipantResponse,
    RoundListResponse,
    RoundResponse,
    StartJobRequest,
    SubmitUpdateRequest,
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


def _get_current_tenant() -> str:
    """Extract tenant_id from request context — override with auth middleware."""
    return "default-tenant"


JobServiceDep = Annotated[Any, Depends(_get_job_service)]
CoordinationServiceDep = Annotated[Any, Depends(_get_coordination_service)]
TrainingServiceDep = Annotated[Any, Depends(_get_training_service)]
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
