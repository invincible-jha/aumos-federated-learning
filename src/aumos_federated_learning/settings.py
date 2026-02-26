"""Settings for aumos-federated-learning via Pydantic BaseSettings."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # Service identity
    service_name: str = Field(default="aumos-federated-learning", alias="AUMOS_SERVICE_NAME")
    service_version: str = Field(default="0.1.0", alias="AUMOS_SERVICE_VERSION")
    env: str = Field(default="development", alias="AUMOS_ENV")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://aumos:aumos@localhost:5432/aumos_federated_learning",
        alias="DATABASE_URL",
    )
    database_pool_size: int = Field(default=10, alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, alias="DATABASE_MAX_OVERFLOW")

    # Kafka
    kafka_bootstrap_servers: str = Field(default="localhost:9092", alias="KAFKA_BOOTSTRAP_SERVERS")
    kafka_consumer_group_id: str = Field(
        default="aumos-federated-learning", alias="KAFKA_CONSUMER_GROUP_ID"
    )

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # External services
    privacy_engine_url: str = Field(default="http://localhost:8001", alias="PRIVACY_ENGINE_URL")
    model_registry_url: str = Field(default="http://localhost:8002", alias="MODEL_REGISTRY_URL")

    # FL settings
    fl_default_strategy: str = Field(default="fedavg", alias="FL_DEFAULT_STRATEGY")
    fl_min_participants: int = Field(default=2, alias="FL_MIN_PARTICIPANTS")
    fl_max_rounds: int = Field(default=100, alias="FL_MAX_ROUNDS")
    fl_round_timeout_seconds: int = Field(default=3600, alias="FL_ROUND_TIMEOUT_SECONDS")
    fl_model_storage_backend: str = Field(default="local", alias="FL_MODEL_STORAGE_BACKEND")
    fl_model_storage_path: str = Field(default="/tmp/aumos_fl_models", alias="FL_MODEL_STORAGE_PATH")

    # Differential privacy defaults
    fl_default_dp_epsilon: float = Field(default=1.0, alias="FL_DEFAULT_DP_EPSILON")
    fl_default_dp_delta: float = Field(default=1e-5, alias="FL_DEFAULT_DP_DELTA")
    fl_dp_noise_multiplier: float = Field(default=1.1, alias="FL_DP_NOISE_MULTIPLIER")
    fl_dp_max_grad_norm: float = Field(default=1.0, alias="FL_DP_MAX_GRAD_NORM")

    # Secure aggregation
    fl_secure_agg_enabled: bool = Field(default=True, alias="FL_SECURE_AGG_ENABLED")
    fl_secure_agg_threshold: float = Field(default=0.6, alias="FL_SECURE_AGG_THRESHOLD")

    # Flower gRPC
    flower_server_address: str = Field(default="0.0.0.0:9091", alias="FLOWER_SERVER_ADDRESS")
    flower_ssl_enabled: bool = Field(default=False, alias="FLOWER_SSL_ENABLED")
    flower_ssl_ca_cert: str = Field(default="", alias="FLOWER_SSL_CA_CERT")
    flower_ssl_server_cert: str = Field(default="", alias="FLOWER_SSL_SERVER_CERT")
    flower_ssl_server_key: str = Field(default="", alias="FLOWER_SSL_SERVER_KEY")

    # Auth
    jwt_secret_key: str = Field(default="change-me-in-production", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=60, alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")

    # Observability
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    sentry_dsn: str = Field(default="", alias="SENTRY_DSN")
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317", alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )

    model_config = {"populate_by_name": True, "env_file": ".env", "extra": "ignore"}


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
