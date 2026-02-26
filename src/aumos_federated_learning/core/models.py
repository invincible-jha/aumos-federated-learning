"""SQLAlchemy domain models for aumos-federated-learning.

All tables use the fed_ prefix and include tenant isolation via TenantMixin.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class TenantMixin:
    """Mixin providing tenant_id column for multi-tenant row-level security."""

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)


class TimestampMixin:
    """Mixin providing created_at and updated_at audit columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class FederatedJob(TenantMixin, TimestampMixin, Base):
    """A federated learning training job coordinating multiple participants.

    Status lifecycle: configuring → recruiting → training → aggregating → complete | failed
    """

    __tablename__ = "fed_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Status lifecycle
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="configuring",
        index=True,
    )

    # FL strategy configuration
    strategy: Mapped[str] = mapped_column(
        String(32), nullable=False, default="fedavg"
    )
    num_rounds: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    current_round: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Participation thresholds
    min_participants: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    actual_participants: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Differential privacy configuration
    dp_epsilon: Mapped[Decimal | None] = mapped_column(
        Numeric(precision=10, scale=6), nullable=True
    )
    dp_delta: Mapped[Decimal | None] = mapped_column(
        Numeric(precision=20, scale=15), nullable=True
    )

    # Output
    aggregated_model_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Strategy-specific hyperparameters (FedProx mu, SCAFFOLD lr, etc.)
    strategy_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Whether fallback to synthetic data was used
    synthetic_fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Relationships
    participants: Mapped[list["Participant"]] = relationship(
        "Participant", back_populates="job", cascade="all, delete-orphan"
    )
    rounds: Mapped[list["AggregationRound"]] = relationship(
        "AggregationRound", back_populates="job", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<FederatedJob id={self.id} name={self.name!r} status={self.status}>"


class Participant(TenantMixin, TimestampMixin, Base):
    """An organization participating in a FederatedJob.

    Status lifecycle: invited → accepted → training → submitted | dropped
    """

    __tablename__ = "fed_participants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("fed_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    organization_name: Mapped[str] = mapped_column(String(256), nullable=False)
    organization_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="invited", index=True
    )

    # Training data metadata (no raw data ever stored here)
    data_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    data_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Contribution tracking
    contribution_weight: Mapped[Decimal | None] = mapped_column(
        Numeric(precision=10, scale=6), nullable=True
    )
    rounds_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Secure aggregation: participant's public key for MPC mask exchange
    public_key_pem: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    job: Mapped["FederatedJob"] = relationship("FederatedJob", back_populates="participants")

    def __repr__(self) -> str:
        return f"<Participant id={self.id} org={self.organization_name!r} status={self.status}>"


class AggregationRound(TimestampMixin, Base):
    """A single training round in a FederatedJob."""

    __tablename__ = "fed_aggregation_rounds"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("fed_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    participants_submitted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    aggregation_method: Mapped[str] = mapped_column(String(32), nullable=False)
    dp_noise_added: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Convergence and training metrics for this round
    round_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Model artifact for this round
    round_model_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Relationships
    job: Mapped["FederatedJob"] = relationship("FederatedJob", back_populates="rounds")

    def __repr__(self) -> str:
        return f"<AggregationRound job={self.job_id} round={self.round_number}>"
