"""Resume / idempotency: keys reload from disk, records are append-only."""
from src.persistence import RunStore
from src.pipeline import (
    _decompose_key,
    _generation_key,
    _judgment_key,
    _verify_key,
)


def test_generation_key_is_deterministic_and_field_sensitive():
    a = _generation_key("openai:m", "doc", 1000, "v1")
    b = _generation_key("openai:m", "doc", 1000, "v1")
    assert a == b
    assert a != _generation_key("openai:m", "doc", 2000, "v1")
    assert a != _generation_key("openai:m", "doc2", 1000, "v1")
    assert a != _generation_key("openai:other", "doc", 1000, "v1")


def test_judgment_and_verify_keys_include_judge():
    gk = _generation_key("m", "d", 1000, "v1")
    assert _judgment_key(gk, "ja", "v1") != _judgment_key(gk, "jb", "v1")
    assert _verify_key(gk, 0, "ja") != _verify_key(gk, 0, "jb")
    assert _verify_key(gk, 0, "ja") != _verify_key(gk, 1, "ja")


def test_generation_resume_reloads_keys(tmp_path):
    store = RunStore("r", str(tmp_path))
    rec = {"idempotency_key": "k1", "model_spec": "m",
           "requested_context_tokens": 1000, "cost_estimate_usd": 0.1}
    assert not store.has_generation("k1")
    store.append_generation(rec)
    assert store.has_generation("k1")

    reopened = RunStore("r", str(tmp_path))  # fresh instance, same dir
    assert reopened.has_generation("k1")
    assert len(reopened.load_generations()) == 1


def test_verification_and_decomposition_resume(tmp_path):
    store = RunStore("r", str(tmp_path))
    store.append_decomposition({"idempotency_key": "d1", "generation_key": "g",
                                "decomposer_spec": "j", "claims": ["a"],
                                "cost_estimate_usd": 0.0})
    store.append_verification({"idempotency_key": "v1", "generation_key": "g",
                               "claim_id": 0, "judge_spec": "j",
                               "verdict": "supported", "cost_estimate_usd": 0.0})
    reopened = RunStore("r", str(tmp_path))
    assert reopened.has_decomposition("d1")
    assert reopened.has_verification("v1")


def test_config_roundtrip(tmp_path):
    store = RunStore("r", str(tmp_path))
    store.save_config({"run_id": "r", "judge_specs": ["a", "b"]})
    assert RunStore("r", str(tmp_path)).load_config()["judge_specs"] == ["a", "b"]
