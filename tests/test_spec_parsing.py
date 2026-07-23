"""Model spec-string parsing via the factory."""
import pytest

from src.backends import create_backend
from src.backends.mock import MockBackend


def test_mock_spec():
    b = create_backend("mock:foo")
    assert isinstance(b, MockBackend)
    assert b.model_id == "foo"
    assert b.provider == "mock"


def test_missing_colon_raises():
    with pytest.raises(ValueError):
        create_backend("nocolon")


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        create_backend("weird:model")


def test_model_id_with_slashes_preserved():
    b = create_backend("mock:Org/Model-Name-7B")
    assert b.model_id == "Org/Model-Name-7B"


def test_openai_query_params(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "k")
    b = create_backend("openai:grok-4.5?base_url=https://api.x.ai/v1&key_env=XAI_API_KEY")
    assert b.provider == "openai"
    assert b.model_id == "grok-4.5"


def test_openai_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        create_backend("openai:gpt-5.5")


def test_anthropic_spec(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    b = create_backend("anthropic:claude-opus-4-8")
    assert b.provider == "anthropic"
    assert b.model_id == "claude-opus-4-8"
