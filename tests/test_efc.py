"""EFC computation on synthetic rate curves."""
from scripts.analyze import compute_efc


def test_efc_at_the_crossing():
    # rate(1000)=0.10, threshold=0.20 -> 4000 (0.15) qualifies, 16000 (0.30) does not
    rates = {1000: 0.10, 4000: 0.15, 16000: 0.30, 64000: 0.60}
    r = compute_efc(rates, baseline_length=1000, delta=0.10)
    assert r["efc"] == 4000
    assert r["baseline_length_used"] == 1000
    assert r["baseline_rate"] == 0.10
    assert abs(r["threshold"] - 0.20) < 1e-9


def test_efc_all_lengths_qualify():
    rates = {1000: 0.10, 2000: 0.11, 4000: 0.12}
    assert compute_efc(rates, 1000, 0.10)["efc"] == 4000


def test_efc_only_baseline_qualifies():
    rates = {1000: 0.10, 2000: 0.50, 4000: 0.90}
    assert compute_efc(rates, 1000, 0.10)["efc"] == 1000


def test_efc_baseline_not_tested_uses_nearest():
    rates = {900: 0.10, 4000: 0.15}
    r = compute_efc(rates, baseline_length=1000, delta=0.10)
    assert r["baseline_length_used"] == 900


def test_efc_largest_qualifying_even_if_non_monotonic():
    # A dip back under threshold at 32000 should be picked as the largest L.
    rates = {1000: 0.10, 8000: 0.40, 32000: 0.18}
    r = compute_efc(rates, 1000, 0.10)  # threshold 0.20
    assert r["efc"] == 32000
