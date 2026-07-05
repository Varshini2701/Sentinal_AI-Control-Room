"""Tests for environment-driven configuration."""

from __future__ import annotations

import pytest

from sentinel.config import Environment, Settings, get_settings, reload_settings
from sentinel.config.settings import DecisionSettings, PostgresSettings


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any SENTINEL_* env vars so tests start from documented defaults."""
    import os

    for key in list(os.environ):
        if key.startswith("SENTINEL_"):
            monkeypatch.delenv(key, raising=False)
    reload_settings()


class TestDefaults:
    def test_default_settings(self) -> None:
        settings = Settings()
        assert settings.environment is Environment.LOCAL
        assert settings.rabbitmq.port == 5672
        assert settings.redis.db == 0
        assert settings.decision.min_green_s == 7.0
        assert settings.is_production is False

    def test_get_settings_is_cached(self) -> None:
        assert get_settings() is get_settings()


class TestEnvOverride:
    def test_scalar_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_ENVIRONMENT", "production")
        monkeypatch.setenv("SENTINEL_SERVICE_NAME", "decision-agent")
        settings = reload_settings()
        assert settings.environment is Environment.PRODUCTION
        assert settings.is_production is True
        assert settings.service_name == "decision-agent"

    def test_nested_override_with_delimiter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_RABBITMQ__HOST", "rabbit.internal")
        monkeypatch.setenv("SENTINEL_RABBITMQ__PORT", "5673")
        monkeypatch.setenv("SENTINEL_DECISION__MIN_GREEN_S", "9")
        settings = reload_settings()
        assert settings.rabbitmq.host == "rabbit.internal"
        assert settings.rabbitmq.port == 5673
        assert settings.decision.min_green_s == 9.0


class TestValidation:
    def test_max_green_below_min_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_green_s must be >= min_green_s"):
            DecisionSettings(min_green_s=30, max_green_s=10)

    def test_starvation_below_max_green_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_starvation_s must be >= max_green_s"):
            DecisionSettings(max_green_s=60, max_starvation_s=30)

    def test_postgres_pool_ordering(self) -> None:
        with pytest.raises(ValueError, match="pool_max_size must be >= pool_min_size"):
            PostgresSettings(pool_min_size=10, pool_max_size=2)

    def test_port_out_of_range_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_REDIS__PORT", "99999")
        with pytest.raises(ValueError):
            reload_settings()


class TestConnectionStrings:
    def test_rabbitmq_url(self) -> None:
        settings = Settings()
        url = settings.rabbitmq.url
        assert url.startswith("amqp://guest:guest@localhost:5672")

    def test_password_is_secret(self) -> None:
        settings = Settings()
        # SecretStr must not leak the value in its repr.
        assert "sentinel" not in repr(settings.postgres.password)
        assert settings.postgres.password.get_secret_value() == "sentinel"

    def test_redis_url_without_password(self) -> None:
        settings = Settings()
        assert settings.redis.url == "redis://localhost:6379/0"
