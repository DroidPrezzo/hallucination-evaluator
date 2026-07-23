"""Truncation exactness at token boundaries."""
import pytest

from src.backends.mock import MockBackend


def test_truncate_lands_on_exact_token_boundary():
    b = MockBackend()
    text = " ".join(f"w{i}" for i in range(1000))
    for target in (1, 5, 50, 333, 999):
        out = b.truncate(text, target)
        assert b.count_tokens(out) == target  # exact, not target±k


def test_truncate_shorter_than_target_is_unchanged():
    b = MockBackend()
    text = "one two three"
    assert b.truncate(text, 10) == text
    assert b.count_tokens(b.truncate(text, 10)) == 3


def test_truncate_equal_to_target_is_unchanged():
    b = MockBackend()
    text = "a b c d e"
    assert b.truncate(text, 5) == text


def test_truncate_is_a_prefix():
    b = MockBackend()
    text = " ".join(f"t{i}" for i in range(100))
    out = b.truncate(text, 10)
    assert text.startswith(out)
    assert out == " ".join(f"t{i}" for i in range(10))


def test_tiktoken_truncation_does_not_exceed_target(monkeypatch):
    tiktoken = pytest.importorskip("tiktoken")
    try:
        tiktoken.get_encoding("cl100k_base")
    except Exception:
        pytest.skip("cl100k_base vocab unavailable offline")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from src.backends.openai_compat import OpenAICompatBackend

    b = OpenAICompatBackend("gpt-4")
    text = "word " * 2000
    out = b.truncate(text, 50)
    n = b.count_tokens(out)
    assert n <= 50
    assert n >= 40  # a real truncation near the target, not over-trimmed
