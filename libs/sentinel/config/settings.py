"""Typed, environment-driven configuration for every Sentinel AI service.

Configuration is layered: nested :class:`pydantic_settings.BaseSettings` groups map to
``SENTINEL_<GROUP>__<FIELD>`` environment variables (double underscore is the nesting
delimiter), with an optional ``.env`` file for local development. :func:`get_settings` returns
a process-wide cached instance so configuration is parsed and validated exactly once.

Example::

    export SENTINEL_ENVIRONMENT=production
    export SENTINEL_RABBITMQ__HOST=rabbit.internal
    export SENTINEL_RABBITMQ__PORT=5672
    export SENTINEL_DECISION__MIN_GREEN_S=7
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Deployment environment; controls logging format and safety of defaults."""

    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class RabbitMqSettings(BaseSettings):
    """Connection settings for the inter-agent event bus."""

    model_config = SettingsConfigDict(env_prefix="SENTINEL_RABBITMQ__")

    host: str = "localhost"
    port: int = Field(default=5672, ge=1, le=65535)
    username: str = "guest"
    password: SecretStr = SecretStr("guest")
    vhost: str = "/"
    exchange: str = "sentinel.events"
    prefetch_count: int = Field(default=32, ge=1)
    connection_timeout_s: float = Field(default=10.0, gt=0)

    @property
    def url(self) -> str:
        """Build the AMQP connection URL (password included; do not log)."""
        pwd = self.password.get_secret_value()
        vhost = self.vhost.lstrip("/")
        return f"amqp://{self.username}:{pwd}@{self.host}:{self.port}/{vhost}"


class RedisSettings(BaseSettings):
    """Connection settings for state storage and high-rate stream deltas."""

    model_config = SettingsConfigDict(env_prefix="SENTINEL_REDIS__")

    host: str = "localhost"
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0)
    password: SecretStr | None = None
    state_ttl_s: int = Field(default=30, ge=1, description="TTL for live IntersectionState keys")
    stream_maxlen: int = Field(
        default=10_000, ge=1, description="Approx cap on Redis stream length (XADD MAXLEN)"
    )

    @property
    def url(self) -> str:
        auth = f":{self.password.get_secret_value()}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class PostgresSettings(BaseSettings):
    """Connection settings for the historical / audit store."""

    model_config = SettingsConfigDict(env_prefix="SENTINEL_POSTGRES__")

    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = "sentinel"
    username: str = "sentinel"
    password: SecretStr = SecretStr("sentinel")
    pool_min_size: int = Field(default=2, ge=1)
    pool_max_size: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def _check_pool(self) -> PostgresSettings:
        if self.pool_max_size < self.pool_min_size:
            raise ValueError("pool_max_size must be >= pool_min_size")
        return self

    @property
    def dsn(self) -> str:
        pwd = self.password.get_secret_value()
        return (
            f"postgresql://{self.username}:{pwd}@{self.host}:{self.port}/{self.database}"
        )


class ObservabilitySettings(BaseSettings):
    """Logging, metrics and tracing configuration."""

    model_config = SettingsConfigDict(env_prefix="SENTINEL_OBS__")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = Field(default=True, description="Emit JSON logs (False = console renderer)")
    metrics_enabled: bool = True
    metrics_port: int = Field(default=9100, ge=1, le=65535)
    tracing_enabled: bool = False
    otlp_endpoint: str | None = None


class DecisionSettings(BaseSettings):
    """Safety envelope and policy parameters for the Decision / Signal Controller agents.

    These constants are the hard, inviolable bounds of the control system. They live in config
    (not code) so a traffic engineer can tune them per intersection without a redeploy, but the
    Signal Controller enforces them regardless of what any policy or LLM proposes.
    """

    model_config = SettingsConfigDict(env_prefix="SENTINEL_DECISION__")

    min_green_s: float = Field(default=7.0, gt=0)
    max_green_s: float = Field(default=60.0, gt=0)
    yellow_s: float = Field(default=3.0, gt=0)
    all_red_s: float = Field(default=2.0, ge=0, description="Mandatory clearance interval")
    pedestrian_min_s: float = Field(default=6.0, gt=0)
    max_starvation_s: float = Field(
        default=120.0, gt=0, description="A lane must be served within this window (fairness)"
    )
    no_movement_switch_s: float = Field(
        default=18.0, gt=0, description="Switch away from a green lane with no movement after this"
    )
    queue_congestion_threshold_m: float = Field(default=15.0, gt=0)
    min_perception_confidence: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Below this, drop to DEGRADED fixed-timer control",
    )

    # --- multi-objective utility-policy weights (tunable, not safety-critical) ---
    weight_queue: float = Field(default=1.0, ge=0.0, description="Weight on queued-vehicle count")
    weight_wait: float = Field(default=0.4, ge=0.0, description="Weight on accumulated wait time")
    weight_fairness: float = Field(
        default=0.15, ge=0.0, description="Weight on time an axis has waited for green"
    )
    weight_pedestrian: float = Field(
        default=10.0,
        ge=0.0,
        description="Utility bump when a pedestrian is waiting (above switch_penalty so a "
        "waiting pedestrian on the opposing axis can justify a switch)",
    )
    weight_prediction: float = Field(
        default=0.5, ge=0.0, description="Weight on predicted future queue (from the forecast)"
    )
    switch_penalty: float = Field(
        default=8.0,
        ge=0.0,
        description="Utility margin the opposing axis must beat to justify a phase switch "
        "(models the cost of the lost clearance interval; prevents thrashing)",
    )

    @model_validator(mode="after")
    def _check_green_bounds(self) -> DecisionSettings:
        if self.max_green_s < self.min_green_s:
            raise ValueError("max_green_s must be >= min_green_s")
        if self.max_starvation_s < self.max_green_s:
            raise ValueError("max_starvation_s must be >= max_green_s")
        return self


class PerceptionSettings(BaseSettings):
    """Perception-pipeline knobs (consumed by the perception worker in a later module)."""

    model_config = SettingsConfigDict(env_prefix="SENTINEL_PERCEPTION__")

    model_path: str = "models/weights/yolo.pt"
    device: str = "cuda:0"
    confidence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    batch_size: int = Field(default=4, ge=1)
    target_fps: float = Field(default=10.0, gt=0)
    stopped_speed_threshold: float = Field(
        default=1.5, ge=0, description="Speed below which a track is 'stopped' (px or m /s)"
    )


class MemorySettings(BaseSettings):
    """Rolling-window history retained by the Traffic Memory Agent (in-process default)."""

    model_config = SettingsConfigDict(env_prefix="SENTINEL_MEMORY__")

    window_size: int = Field(
        default=3600, ge=1, description="Max state snapshots retained per intersection"
    )
    baseline_publish_every: int = Field(
        default=20, ge=1, description="Publish a refreshed baseline every N state updates"
    )


class PredictionSettings(BaseSettings):
    """Short-horizon forecasting parameters for the Prediction Agent."""

    model_config = SettingsConfigDict(env_prefix="SENTINEL_PREDICTION__")

    horizon_s: float = Field(default=60.0, gt=0, description="How far ahead to forecast")
    trend_window: int = Field(
        default=12, ge=2, description="Recent samples used to fit the linear trend"
    )
    min_samples_for_trend: int = Field(
        default=4, ge=2, description="Below this, fall back to the naive persistence forecast"
    )


class IncidentSettings(BaseSettings):
    """Thresholds for the Incident Detection Agent's rule set."""

    model_config = SettingsConfigDict(env_prefix="SENTINEL_INCIDENT__")

    stalled_wait_s: float = Field(
        default=90.0, gt=0, description="Stopped-in-a-served-lane duration flagged as stalled"
    )
    congestion_ratio: float = Field(
        default=2.5, gt=1.0, description="Queue / baseline-avg-queue ratio flagged as abnormal"
    )
    congestion_min_baseline_samples: int = Field(
        default=10, ge=1, description="Minimum baseline samples before congestion checks apply"
    )
    debounce_s: float = Field(
        default=60.0, ge=0, description="Minimum time before re-raising the same incident"
    )


class ExplainabilitySettings(BaseSettings):
    """Configuration for the Explainability Agent. The LLM narrates; it never decides."""

    model_config = SettingsConfigDict(env_prefix="SENTINEL_EXPLAIN__")

    use_llm: bool = Field(
        default=False, description="Use the LLM narrator; falls back to templates on failure"
    )
    model_id: str = "claude-sonnet-5"
    timeout_s: float = Field(default=8.0, gt=0)
    max_tokens: int = Field(default=200, ge=1)


class OrchestratorSettings(BaseSettings):
    """Health-watchdog and mode-switching parameters for the Orchestrator Agent."""

    model_config = SettingsConfigDict(env_prefix="SENTINEL_ORCHESTRATOR__")

    stale_after_s: float = Field(
        default=15.0, gt=0, description="An agent with no heartbeat this long is marked unhealthy"
    )
    check_interval_s: float = Field(
        default=5.0, ge=0, description="Staleness check cadence; 0 disables the watchdog loop"
    )


class Settings(BaseSettings):
    """Root application settings aggregating every configuration group."""

    model_config = SettingsConfigDict(
        env_prefix="SENTINEL_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = Environment.LOCAL
    service_name: str = Field(default="sentinel", min_length=1)
    intersection_id: str = Field(default="intersection-1", min_length=1)

    rabbitmq: RabbitMqSettings = Field(default_factory=RabbitMqSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    decision: DecisionSettings = Field(default_factory=DecisionSettings)
    perception: PerceptionSettings = Field(default_factory=PerceptionSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    prediction: PredictionSettings = Field(default_factory=PredictionSettings)
    incident: IncidentSettings = Field(default_factory=IncidentSettings)
    explainability: ExplainabilitySettings = Field(default_factory=ExplainabilitySettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings instance.

    Cached so that env parsing and validation happen once. Call :func:`reload_settings` in tests
    after mutating the environment to force a re-read.
    """
    return Settings()


def reload_settings() -> Settings:
    """Clear the settings cache and return a freshly parsed instance (test helper)."""
    get_settings.cache_clear()
    return get_settings()
